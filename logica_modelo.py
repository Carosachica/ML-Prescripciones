"""Lógica de puntuación del modelo de priorización de cobro coactivo (FFDS).

Mismas funciones del Notebook 2, separadas de la interfaz para poder reutilizarlas y probarlas.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUTA_MODELO = Path(__file__).parent / 'modelo' / 'modelo_priorizacion_ffds.joblib'
RUTA_DEMO = Path(__file__).parent / 'modelo' / 'demo_expedientes_nuevos.csv'
RUTA_CARTERA = Path(__file__).parent / 'modelo' / 'cartera_vigente_priorizada.csv'
RUTA_CURVA = Path(__file__).parent / 'modelo' / 'curva_ganancia.csv'
COLOR = {'Alta': '#C8102E', 'Media': '#E89B0C', 'Baja': '#8794A6'}
AZUL, VERDE, GRIS, TINTA = '#0B2545', '#0E9F8E', '#D9DFE7', '#1B2A41'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': '#C9D1DB', 'axes.labelcolor': '#4A5568', 'xtick.color': '#4A5568', 'ytick.color': '#4A5568',
                     'axes.titleweight': 'bold', 'axes.titlecolor': TINTA, 'axes.titlesize': 11, 'figure.dpi': 150})


def cargar_paquete(ruta=RUTA_MODELO):
    return joblib.load(ruta)


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p)).reshape(-1, 1)


def catalogos(paquete, segmento):
    """Opciones válidas para los campos categóricos del formulario."""
    seg = paquete['segmentos'][segmento]
    quitar = lambda pref: [c.replace(pref, '') for c in seg['variables'] if c.startswith(pref)]
    return {'areas': quitar('area_generadora_') or ['Otros'], 'tipos_id': quitar('tipo_id_') or ['NI'], 'actos': sorted(seg['mapa_actos']) or ['mandamiento_pago']}


def puntuar(paquete, X, segmento):
    """X: DataFrame con las variables del modelo. Devuelve probabilidad, componentes, puntaje y prioridad."""
    seg, pesos, cortes = paquete['segmentos'][segmento], paquete['pesos_puntaje'], paquete['cortes']
    ref = seg['referencia']
    X = X.reindex(columns=seg['variables']).fillna(0.0)
    prob = seg['calibrador'].predict_proba(_logit(seg['modelo'].predict_proba(X.values)[:, 1]))[:, 1]
    saldo, dias, n = np.round(np.expm1(X['log1p_saldo_pos'].values)), X['dias_a_prescripcion'].values, len(ref['prob'])
    out = pd.DataFrame({'probabilidad': prob, 'saldo': saldo, 'dias_a_prescripcion': dias,
                        'pct_probabilidad': np.searchsorted(ref['prob'], np.round(prob, 10), side='right') / n,
                        'pct_valor': np.searchsorted(ref['saldo'], saldo, side='right') / n,
                        'pct_urgencia': 1 - np.searchsorted(ref['dias'], dias, side='left') / n}, index=X.index)
    out['puntaje'] = 100 * (pesos['prob'] * out.pct_probabilidad + pesos['valor'] * out.pct_valor + pesos['urgencia'] * out.pct_urgencia)
    out['prioridad'] = np.where(out.puntaje >= cortes['alta'], 'Alta', np.where(out.puntaje >= cortes['media'], 'Media', 'Baja'))
    out['veces_tasa_base'] = out.probabilidad / seg['tasa_base']
    return out


def construir_fila(paquete, segmento, dias_a_prescripcion, dias_desde_ultimo_acto, valor_multa, pagado_capital=0.0, pagado_interes=0.0, anio_origen=None,
                   area_generadora=None, ultimo_acto=None, tipo_identificacion=None, otros_exp_misma_direccion=0, otro_con_pago_misma_direccion=False,
                   otros_codigos_mismo_expediente=0, otro_codigo_con_pago=False):
    """Convierte los datos de un expediente en la fila de variables que espera el modelo."""
    seg = paquete['segmentos'][segmento]
    fila = dict(seg['valores_tipicos'])
    saldo, pagado = valor_multa - pagado_capital, pagado_capital + pagado_interes
    num = {'dias_a_prescripcion': dias_a_prescripcion, 'dias_desde_ultimo_acto': dias_desde_ultimo_acto, 'log1p_valor_final': np.log1p(max(valor_multa, 0)),
           'log1p_pagado_total': np.log1p(max(pagado, 0)), 'log1p_pagado_capital': np.log1p(max(pagado_capital, 0)), 'log1p_pagado_interes': np.log1p(max(pagado_interes, 0)),
           'log1p_saldo_pos': np.log1p(max(saldo, 0)), 'pct_pagado': float(np.clip(pagado / valor_multa, 0, 1)) if valor_multa > 0 else 0.0,
           'ratio_saldo_valor': float(np.clip(max(saldo, 0) / valor_multa, 0, 1.5)) if valor_multa > 0 else None,
           'log1p_n_otros_misma_direccion': np.log1p(otros_exp_misma_direccion), 'log1p_n_otros_mismo_expediente': np.log1p(otros_codigos_mismo_expediente), 'anio_origen': anio_origen}
    for c, v in num.items():
        if c in fila and v is not None:
            lo, hi = seg['limites_winsor'].get(c, (-np.inf, np.inf))
            fila[c] = float(np.clip(v, lo, hi))
    binarias = {'saldo_negativo': saldo < 0, 'interes_negativo': pagado_interes < 0, 'tiene_pago': pagado > 0, 'prescripcion_imputada': False,
                'saldo_residual': 0 < saldo < paquete['umbral_saldo_residual'], 'otros_con_pago_misma_direccion': otro_con_pago_misma_direccion,
                'otros_con_pago_mismo_expediente': otro_codigo_con_pago}
    for c, v in binarias.items():
        if c in fila:
            fila[c] = int(bool(v))
    for c in fila:
        if c.endswith('__nulo'):
            fila[c] = 0
    acto = seg['mapa_actos'].get(ultimo_acto)
    if acto:
        for c, k in (('frec_acto_canonico', 'frec'), ('te_acto_canonico', 'te')):
            if c in fila:
                fila[c] = acto[k]
    for prefijo, valor in (('area_generadora_', area_generadora), ('tipo_id_', tipo_identificacion), ('fase_procesal_', acto['fase'] if acto else None)):
        cols = [c for c in fila if c.startswith(prefijo)]
        if cols and valor is not None:
            for c in cols:
                fila[c] = 0
            destino = prefijo + str(valor) if prefijo + str(valor) in fila else (prefijo + 'Otros' if prefijo + 'Otros' in fila else None)
            if destino:
                fila[destino] = 1
    return pd.DataFrame([fila])[seg['variables']]


def _limpiar(ax):
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.grid(axis='y', color='#EEF1F5', lw=0.8)
    ax.set_axisbelow(True)


def cargar_cartera(ruta=RUTA_CARTERA):
    return pd.read_csv(ruta, dtype={'id_expediente': str})


def cargar_curva(ruta=RUTA_CURVA):
    return pd.read_csv(ruta)


def ganancia(curva, segmentos):
    """Curvas acumuladas (fracción de cartera gestionada → fracción de favorables y de valor pagado) por tipo de orden."""
    d = curva[curva.segmento.isin(segmentos)].groupby(['orden', 'decil'])[['expedientes', 'favorables', 'valor_pagado']].sum()
    out = {}
    for o in ('modelo', 'fecha_prescripcion'):
        t = d.loc[o].sort_index()
        out[o] = {'x': np.r_[0, t.expedientes.cumsum() / t.expedientes.sum()], 'fav': np.r_[0, t.favorables.cumsum() / t.favorables.sum()],
                  'val': np.r_[0, t.valor_pagado.cumsum() / t.valor_pagado.sum()]}
    return out


def captura(g, orden, capacidad, cual='fav'):
    return float(np.interp(capacidad, g[orden]['x'], g[orden][cual]))


def figura_ganancia(g, capacidad):
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ax.fill_between(g['modelo']['x'] * 100, g['modelo']['fav'] * 100, g['fecha_prescripcion']['fav'] * 100, color=VERDE, alpha=0.10)
    ax.plot(g['modelo']['x'] * 100, g['modelo']['fav'] * 100, color=VERDE, lw=3, label='Orden del modelo')
    ax.plot(g['fecha_prescripcion']['x'] * 100, g['fecha_prescripcion']['fav'] * 100, color=AZUL, lw=2, ls='--', label='Orden por fecha de prescripción')
    ax.plot([0, 100], [0, 100], color='#B6BFCB', lw=1.2, ls=':', label='Sin priorizar')
    ym, yr = captura(g, 'modelo', capacidad) * 100, captura(g, 'fecha_prescripcion', capacidad) * 100
    ax.axvline(capacidad * 100, color='#B6BFCB', lw=1)
    ax.scatter([capacidad * 100] * 2, [ym, yr], color=[VERDE, AZUL], s=60, zorder=5)
    ax.annotate(f'{ym:.0f} %', (capacidad * 100, ym), xytext=(8, 4), textcoords='offset points', color=VERDE, fontweight='bold', fontsize=12)
    ax.annotate(f'{yr:.0f} %', (capacidad * 100, yr), xytext=(8, -14), textcoords='offset points', color=AZUL, fontweight='bold', fontsize=11)
    ax.set_xlim(0, 100); ax.set_ylim(0, 102); ax.set_xlabel('% de la cartera que el equipo alcanza a gestionar'); ax.set_ylabel('% de desenlaces favorables cubiertos')
    ax.legend(frameon=False, loc='lower right', fontsize=9); _limpiar(ax); fig.tight_layout()
    return fig


def figura_posicion(paquete, res, segmento):
    """Posición del expediente frente a la cartera vigente de su segmento."""
    ref, cortes, col = paquete['segmentos'][segmento]['referencia'], paquete['cortes'], COLOR[res.prioridad]
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    n, bins, parches = ax.hist(ref['puntaje'], bins=45, color=GRIS)
    for b, p in zip(bins[:-1], parches):
        p.set_facecolor(COLOR['Alta'] if b >= cortes['alta'] else COLOR['Media'] if b >= cortes['media'] else COLOR['Baja']); p.set_alpha(0.35)
    ax.axvline(res.puntaje, color=col, lw=3)
    ax.annotate('Este expediente', (res.puntaje, n.max()), xytext=(-6 if res.puntaje > 60 else 6, -2), textcoords='offset points', ha='right' if res.puntaje > 60 else 'left',
                va='top', color=col, fontweight='bold')
    ax.set_xlim(0, 100); ax.set_xlabel('Puntaje de priorización'); ax.set_ylabel('Expedientes vigentes'); _limpiar(ax); fig.tight_layout()
    return fig


def figura_sensibilidad(paquete, segmento, dias_actual, args):
    """Probabilidad y puntaje del mismo expediente al cambiar solo los días que faltan para prescribir."""
    ref, cortes = paquete['segmentos'][segmento]['referencia'], paquete['cortes']
    dias = np.unique(np.linspace(30, np.percentile(ref['dias'], 99), 40).astype(int))
    sim = puntuar(paquete, pd.concat([construir_fila(paquete, segmento, dias_a_prescripcion=d, **args) for d in dias], ignore_index=True), segmento)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.2))
    a1.plot(dias, sim.probabilidad * 100, color=VERDE, lw=2.5); a1.axvline(dias_actual, color=AZUL, ls='--', lw=1.2)
    a1.set_title('Probabilidad de desenlace favorable (%)', loc='left'); a1.set_xlabel('Días que faltan para prescribir')
    a2.axhspan(cortes['alta'], 100, color=COLOR['Alta'], alpha=0.08); a2.axhspan(cortes['media'], cortes['alta'], color=COLOR['Media'], alpha=0.08)
    a2.plot(dias, sim.puntaje, color=AZUL, lw=2.5); a2.axvline(dias_actual, color=AZUL, ls='--', lw=1.2, label='Expediente digitado')
    a2.set_ylim(0, 100); a2.set_title('Puntaje de priorización', loc='left'); a2.set_xlabel('Días que faltan para prescribir'); a2.legend(frameon=False, fontsize=9)
    _limpiar(a1); _limpiar(a2); fig.tight_layout()
    return fig


def figura_dispersion(pri):
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    for p in ('Baja', 'Media', 'Alta'):
        d = pri[pri.prioridad == p]
        ax.scatter(d.saldo.clip(lower=1000), d.probabilidad * 100, s=26, color=COLOR[p], alpha=0.75, label=p, edgecolor='white', linewidth=0.4)
    ax.set_xscale('log'); ax.set_xlabel('Saldo a capital (COP, escala logarítmica)'); ax.set_ylabel('Probabilidad de desenlace favorable (%)')
    h_, l_ = ax.get_legend_handles_labels(); ax.legend(h_[::-1], l_[::-1], title='Prioridad', frameon=False, fontsize=9); _limpiar(ax); fig.tight_layout()
    return fig


def resumen_prioridad(df):
    """Expedientes, saldo y probabilidad media por prioridad (para la cartera vigente o un lote)."""
    g = df.groupby('prioridad').agg(expedientes=('puntaje', 'size'), saldo=('saldo', 'sum'), prob=('probabilidad', 'mean')).reindex(['Alta', 'Media', 'Baja']).fillna(0)
    g['pct_exp'] = g.expedientes / max(g.expedientes.sum(), 1); g['pct_saldo'] = g.saldo / max(g.saldo.sum(), 1)
    return g


def puntuar_lote(paquete, lote):
    """lote: DataFrame con columnas id_expediente, segmento y las variables preparadas. Devuelve la lista priorizada."""
    faltan = {'id_expediente', 'segmento'} - set(lote.columns)
    if faltan:
        raise ValueError(f'Al archivo le faltan las columnas: {sorted(faltan)}')
    partes = []
    for s in lote['segmento'].unique():
        if s not in paquete['segmentos']:
            raise ValueError(f'Segmento no reconocido: {s!r}. Use "natural" o "juridica".')
        d = lote[lote.segmento == s]
        sin = [c for c in paquete['segmentos'][s]['variables'] if c not in d.columns]
        if len(sin) > 3:
            raise ValueError(f'El archivo no trae {len(sin)} variables del modelo para el segmento {s} (por ejemplo {sin[:3]}). Debe ser un archivo preparado por el notebook de preparación.')
        r = puntuar(paquete, d, s)
        r.insert(0, 'segmento', s); r.insert(0, 'id_expediente', d['id_expediente'].astype(str).values)
        partes.append(r)
    out = pd.concat(partes).sort_values('puntaje', ascending=False).reset_index(drop=True)
    out.index += 1
    out.index.name = 'orden'
    return out
