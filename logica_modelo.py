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
COLOR = {'Alta': '#c0392b', 'Media': '#e67e22', 'Baja': '#7f8c8d'}


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


def figura_tarjeta(paquete, res, segmento):
    """Componentes del puntaje y posición del expediente frente a la cartera vigente."""
    seg, pesos, cortes = paquete['segmentos'][segmento], paquete['pesos_puntaje'], paquete['cortes']
    ref, col = seg['referencia'], COLOR[res.prioridad]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.8))
    comp = [('Probabilidad de\ndesenlace favorable', res.pct_probabilidad, pesos['prob'], f'{res.probabilidad * 100:.1f} %'),
            ('Valor\n(saldo a capital)', res.pct_valor, pesos['valor'], f'${res.saldo:,.0f}'),
            ('Urgencia\n(tiempo para prescribir)', res.pct_urgencia, pesos['urgencia'], f'{res.dias_a_prescripcion:,.0f} días')]
    for i, (nom, p, w, txt) in enumerate(comp[::-1]):
        a1.barh(i, 100, color='#ecf0f1')
        a1.barh(i, p * 100, color=col, alpha=0.85)
        a1.text(101, i, f'{txt}\naporta {100 * w * p:.0f} de {100 * w:.0f} pts', va='center', fontsize=9)
    a1.set_yticks(range(3)); a1.set_yticklabels([c[0] for c in comp[::-1]], fontsize=9)
    a1.set_xlim(0, 150); a1.set_xticks([0, 50, 100]); a1.set_xlabel('Posición frente a la cartera vigente (percentil)')
    a1.set_title('¿De dónde sale el puntaje?', fontsize=11, fontweight='bold')
    a2.hist(ref['puntaje'], bins=40, color='#bdc3c7')
    a2.axvspan(cortes['alta'], 100, color=COLOR['Alta'], alpha=0.12); a2.axvspan(cortes['media'], cortes['alta'], color=COLOR['Media'], alpha=0.12)
    a2.axvline(res.puntaje, color=col, lw=3)
    a2.set_title(f'Supera al {(ref["puntaje"] < res.puntaje).mean() * 100:.0f} % de los {len(ref["puntaje"]):,} expedientes vigentes', fontsize=11, fontweight='bold')
    a2.set_xlabel('Puntaje de priorización (zonas: Baja · Media · Alta)'); a2.set_ylabel('Expedientes vigentes')
    for a in (a1, a2):
        for sp in ('top', 'right'):
            a.spines[sp].set_visible(False)
    fig.tight_layout()
    return fig


def figura_sensibilidad(paquete, segmento, dias_actual, args):
    """Probabilidad y puntaje del mismo expediente al cambiar solo los días que faltan para prescribir."""
    ref, cortes = paquete['segmentos'][segmento]['referencia'], paquete['cortes']
    dias = np.unique(np.linspace(30, np.percentile(ref['dias'], 99), 40).astype(int))
    sim = puntuar(paquete, pd.concat([construir_fila(paquete, segmento, dias_a_prescripcion=d, **args) for d in dias], ignore_index=True), segmento)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.4))
    a1.plot(dias, sim.probabilidad * 100, color='#2c3e50', lw=2); a1.axvline(dias_actual, color='#c0392b', ls='--')
    a1.set_title('Probabilidad de desenlace favorable (%)', fontweight='bold'); a1.set_xlabel('Días que faltan para prescribir')
    a2.plot(dias, sim.puntaje, color='#2c3e50', lw=2); a2.axvline(dias_actual, color='#c0392b', ls='--', label='expediente digitado')
    a2.axhspan(cortes['alta'], 100, color=COLOR['Alta'], alpha=0.12); a2.axhspan(cortes['media'], cortes['alta'], color=COLOR['Media'], alpha=0.12)
    a2.set_ylim(0, 100); a2.set_title('Puntaje de priorización', fontweight='bold'); a2.set_xlabel('Días que faltan para prescribir'); a2.legend()
    for a in (a1, a2):
        for sp in ('top', 'right'):
            a.spines[sp].set_visible(False)
    fig.tight_layout()
    return fig


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


def figura_lote(pri):
    orden = ['Alta', 'Media', 'Baja']
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    conteo = pri.groupby('prioridad').size().reindex(orden).fillna(0)
    a1.bar(conteo.index, conteo.values, color=[COLOR[p] for p in conteo.index]); a1.set_title('Expedientes por prioridad', fontweight='bold')
    for p in orden:
        d = pri[pri.prioridad == p]
        a2.scatter(d.saldo.clip(lower=1000), d.probabilidad * 100, s=22, color=COLOR[p], alpha=0.7, label=p)
    a2.set_xscale('log'); a2.set_xlabel('Saldo a capital (COP, escala logarítmica)'); a2.set_ylabel('Probabilidad de desenlace favorable (%)')
    a2.set_title('Probabilidad y valor de cada expediente', fontweight='bold'); a2.legend(title='Prioridad')
    for a in (a1, a2):
        for sp in ('top', 'right'):
            a.spines[sp].set_visible(False)
    fig.tight_layout()
    return fig
