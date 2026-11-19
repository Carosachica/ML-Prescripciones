"""Aplicación Streamlit: priorización de la cartera de cobro coactivo del FFDS.

Ejecutar desde la carpeta del proyecto:   streamlit run app.py
"""
import datetime as dt
import pandas as pd
import sklearn
import streamlit as st
import matplotlib.pyplot as plt
import logica_modelo as lm

st.set_page_config(page_title='Priorización de cobro coactivo · FFDS', page_icon='⚖️', layout='wide', initial_sidebar_state='collapsed')

# ============================================================================ estilo
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stButton, .stTabs { font-family: 'Inter', 'Segoe UI', system-ui, sans-serif; }
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1240px; }
#MainMenu, footer { visibility: hidden; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid #E3E8EF; }
.stTabs [data-baseweb="tab"] { font-weight: 600; font-size: 0.98rem; padding: 10px 18px; color: #5A6779; }
.stTabs [aria-selected="true"] { color: #0B2545; }
.hero { background: linear-gradient(120deg, #0B2545 0%, #13315C 55%, #0E9F8E 130%); border-radius: 18px; padding: 30px 36px 28px; color: #fff; margin-bottom: 18px; }
.hero .sup { font-size: 0.78rem; letter-spacing: 0.14em; text-transform: uppercase; opacity: 0.75; font-weight: 600; }
.hero h1 { font-size: 2.05rem; line-height: 1.15; font-weight: 800; margin: 8px 0 8px; color: #fff; padding: 0; }
.hero p { font-size: 1.05rem; opacity: 0.9; margin: 0 0 16px; max-width: 760px; }
.chip { display: inline-block; background: rgba(255,255,255,0.14); border: 1px solid rgba(255,255,255,0.22); border-radius: 999px; padding: 4px 13px; font-size: 0.8rem; margin-right: 8px; font-weight: 500; }
.kpi { background: #fff; border: 1px solid #E3E8EF; border-radius: 14px; padding: 18px 20px 16px; box-shadow: 0 1px 2px rgba(16,24,40,0.04); min-height: 132px; }
.kpi .lab { font-size: 0.74rem; letter-spacing: 0.08em; text-transform: uppercase; color: #6B7789; font-weight: 600; }
.kpi .val { font-size: 1.95rem; font-weight: 800; color: #0B2545; line-height: 1.15; margin-top: 6px; }
.kpi .sub { font-size: 0.86rem; color: #5A6779; margin-top: 4px; }
.kpi.acento { border-top: 4px solid #0E9F8E; }
.kpi.alta { border-top: 4px solid #C8102E; } .kpi.media { border-top: 4px solid #E89B0C; } .kpi.baja { border-top: 4px solid #8794A6; }
.sec { font-size: 1.22rem; font-weight: 700; color: #0B2545; margin: 30px 0 2px; }
.secsub { color: #5A6779; font-size: 0.94rem; margin-bottom: 12px; }
.cmp { background: #F6F8FB; border-radius: 14px; padding: 18px 20px; margin-bottom: 12px; }
.cmp .t { font-size: 0.8rem; color: #5A6779; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; }
.cmp .fila { display: flex; align-items: baseline; gap: 14px; margin-top: 8px; }
.cmp .g { font-size: 2.2rem; font-weight: 800; color: #0E9F8E; line-height: 1; }
.cmp .vs { font-size: 0.92rem; color: #5A6779; }
.cmp .vs b { color: #0B2545; }
.barra { height: 10px; border-radius: 999px; background: #E9EDF2; overflow: hidden; margin: 8px 0 4px; }
.barra > div { height: 100%; border-radius: 999px; }
.res { background: #fff; border: 1px solid #E3E8EF; border-radius: 16px; padding: 24px 26px; display: flex; gap: 26px; align-items: center; box-shadow: 0 1px 3px rgba(16,24,40,0.06); }
.anillo { width: 148px; height: 148px; border-radius: 50%; display: flex; align-items: center; justify-content: center; flex: none; }
.anillo > div { width: 116px; height: 116px; border-radius: 50%; background: #fff; display: flex; flex-direction: column; align-items: center; justify-content: center; }
.anillo .n { font-size: 2.6rem; font-weight: 800; line-height: 1; }
.anillo .d { font-size: 0.72rem; color: #6B7789; margin-top: 3px; }
.badge { display: inline-block; color: #fff; font-weight: 700; letter-spacing: 0.06em; border-radius: 8px; padding: 5px 14px; font-size: 0.9rem; }
.res h3 { margin: 10px 0 4px; font-size: 1.3rem; color: #0B2545; padding: 0; }
.res p { margin: 0; color: #46536A; font-size: 0.97rem; }
.accion { margin-top: 12px; background: #F6F8FB; border-radius: 10px; padding: 10px 14px; font-size: 0.93rem; color: #1B2A41; }
.comp { background: #fff; border: 1px solid #E3E8EF; border-radius: 16px; padding: 20px 24px; }
.comp .f { margin-bottom: 14px; } .comp .f:last-child { margin-bottom: 0; }
.comp .c1 { display: flex; justify-content: space-between; font-size: 0.93rem; color: #1B2A41; font-weight: 600; }
.comp .c2 { font-size: 0.82rem; color: #6B7789; }
.nota { font-size: 0.82rem; color: #6B7789; margin-top: 8px; }
.check { display: flex; gap: 12px; align-items: flex-start; padding: 12px 0; border-bottom: 1px solid #EEF1F5; }
.check .i { width: 26px; height: 26px; border-radius: 50%; color: #fff; font-weight: 700; display: flex; align-items: center; justify-content: center; flex: none; font-size: 0.85rem; }
.check .tx b { color: #0B2545; } .check .tx { font-size: 0.95rem; color: #46536A; }
</style>
"""


def html(codigo):
    """Inserta HTML sin sangrías ni líneas en blanco (Markdown las interpretaría como bloque de código)."""
    st.markdown(' '.join(linea.strip() for linea in codigo.strip().splitlines() if linea.strip()), unsafe_allow_html=True)


def miles(n):
    return f'{n:,.0f}'.replace(',', '.')


def pesos(v):
    if abs(v) >= 1e9:
        return '$ ' + f'{v / 1e9:,.1f}'.replace(',', 'X').replace('.', ',').replace('X', '.') + ' mil M'
    if abs(v) >= 1e6:
        return '$ ' + f'{v / 1e6:,.1f}'.replace(',', 'X').replace('.', ',').replace('X', '.') + ' M'
    return '$ ' + miles(v)


def pct(v, dec=0):
    return f'{v * 100:.{dec}f}'.replace('.', ',') + ' %'


def num(v, dec=1):
    return f'{v:.{dec}f}'.replace('.', ',')


def kpi(etiqueta, valor, detalle='', clase=''):
    return f'<div class="kpi {clase}"><div class="lab">{etiqueta}</div><div class="val">{valor}</div><div class="sub">{detalle}</div></div>'


def seccion(titulo, subtitulo=''):
    html(f'<div class="sec">{titulo}</div><div class="secsub">{subtitulo}</div>')


def tarjetas_prioridad(resumen):
    """Tres tarjetas (Alta, Media, Baja) con expedientes, saldo y probabilidad media."""
    cols = st.columns(3)
    for c, p in zip(cols, ['Alta', 'Media', 'Baja']):
        r = resumen.loc[p]
        with c:
            html(f'''<div class="kpi {p.lower()}"><div class="lab">Prioridad {p}</div>
                 <div class="val">{miles(r.expedientes)} <span style="font-size:1rem;font-weight:600;color:#5A6779">expedientes</span></div>
                 <div class="barra"><div style="width:{r.pct_exp * 100:.1f}%;background:{lm.COLOR[p]}"></div></div>
                 <div class="sub">{pct(r.pct_exp)} de los expedientes · {pct(r.pct_saldo)} del saldo</div>
                 <div class="sub" style="margin-top:10px"><b style="color:#0B2545">{pesos(r.saldo)}</b> de saldo a capital<br>
                 Probabilidad media de desenlace favorable: <b style="color:#0B2545">{pct(r.prob, 1)}</b></div></div>''')


def tabla_orden(df, n=None):
    v = df[['id_expediente', 'segmento', 'prioridad', 'puntaje', 'probabilidad', 'saldo', 'dias_a_prescripcion']].copy()
    v['probabilidad'] = v['probabilidad'] * 100
    v['segmento'] = v['segmento'].map({'natural': 'Natural', 'juridica': 'Jurídica'}).fillna(v['segmento'])
    if n:
        v = v.head(n)
    st.dataframe(v, hide_index=True, height=min(38 * (len(v) + 1) + 4, 420), column_config={
        'id_expediente': st.column_config.TextColumn('Expediente'), 'segmento': st.column_config.TextColumn('Tipo de persona'),
        'prioridad': st.column_config.TextColumn('Prioridad'),
        'puntaje': st.column_config.ProgressColumn('Puntaje', min_value=0, max_value=100, format='%.0f'),
        'probabilidad': st.column_config.NumberColumn('Prob. favorable', format='%.1f %%'),
        'saldo': st.column_config.NumberColumn('Saldo a capital', format='$ %d'),
        'dias_a_prescripcion': st.column_config.NumberColumn('Días para prescribir', format='%d')})


# ============================================================================ datos
@st.cache_resource
def paquete_modelo():
    return lm.cargar_paquete()


@st.cache_data
def datos_portada():
    return lm.cargar_cartera(), lm.cargar_curva()


st.markdown(CSS, unsafe_allow_html=True)
try:
    PAQ = paquete_modelo()
except Exception as e:                                   # archivo ausente o versión incompatible de scikit-learn
    st.error('No se pudo cargar el modelo (modelo/modelo_priorizacion_ffds.joblib). Ejecute el Notebook 1 en este mismo entorno para generarlo.')
    st.exception(e)
    st.stop()

PESOS, CORTES = PAQ['pesos_puntaje'], PAQ['cortes']
FECHA_CORTE = dt.date.fromisoformat(PAQ['segmentos']['natural']['fecha_corte'][:10]).strftime('%d/%m/%Y')
NOMBRE_SEG = {'natural': 'Personas naturales', 'juridica': 'Personas jurídicas'}
ACCION = {'Alta': 'Incluir en el próximo reparto mensual y gestionar de inmediato.',
          'Media': 'Programar su gestión después de los expedientes de prioridad Alta.',
          'Baja': 'Mantener en gestión ordinaria y volver a evaluar con el siguiente corte de información.'}

with st.sidebar:
    st.subheader('Acerca del modelo')
    st.caption(f"Versión {PAQ['version']} · entrenado el {PAQ['fecha_entrenamiento']}")
    st.write(f"Puntaje = {PESOS['prob'] * 100:.0f} % probabilidad + {PESOS['valor'] * 100:.0f} % valor + {PESOS['urgencia'] * 100:.0f} % urgencia")
    st.write(f"Alta ≥ {CORTES['alta']} · Media ≥ {CORTES['media']}")
    if PAQ['versiones']['scikit-learn'] != sklearn.__version__:
        st.warning(f"El modelo se guardó con scikit-learn {PAQ['versiones']['scikit-learn']} y aquí hay {sklearn.__version__}.")

html(f'''<div class="hero"><div class="sup">Fondo Financiero Distrital de Salud · Cobro coactivo</div>
     <h1>¿Qué expedientes gestionar primero?</h1>
     <p>Un modelo predictivo ordena la cartera según la probabilidad de recuperación, el valor en juego y la cercanía a la prescripción,
     para que la capacidad del equipo se use donde más rinde.</p>
     <span class="chip">Cartera al {FECHA_CORTE}</span><span class="chip">Modelo v{PAQ['version']}</span>
     <span class="chip">Recomienda el orden; la decisión jurídica es del abogado</span></div>''')

tab_res, tab_uno, tab_lote, tab_modelo = st.tabs(['Resumen ejecutivo', 'Evaluar un expediente', 'Priorizar un lote', 'Sobre el modelo'])

# ============================================================================ resumen ejecutivo
with tab_res:
    try:
        CARTERA, CURVA = datos_portada()
    except Exception as e:
        st.error(f'No se encontraron los archivos de la portada en modelo/ (cartera_vigente_priorizada.csv y curva_ganancia.csv). Ejecute el Notebook 1 para generarlos. Detalle: {e}')
    else:
        alcance = st.radio('Cartera', ['Toda la cartera', 'Personas naturales', 'Personas jurídicas'], horizontal=True, label_visibility='collapsed')
        segs = {'Toda la cartera': ['natural', 'juridica'], 'Personas naturales': ['natural'], 'Personas jurídicas': ['juridica']}[alcance]
        car = CARTERA[CARTERA.segmento.isin(segs)]
        res = lm.resumen_prioridad(car)
        g = lm.ganancia(CURVA, segs)
        veces = lm.captura(g, 'modelo', 0.10) / 0.10
        c = st.columns(4)
        with c[0]: html(kpi('Expedientes vigentes priorizados', miles(len(car)), f'Cartera abierta al {FECHA_CORTE}'))
        with c[1]: html(kpi('Saldo a capital en cartera', pesos(car.saldo.sum()), 'Suma de saldos de los expedientes vigentes'))
        with c[2]: html(kpi('Prioridad Alta', miles(res.loc['Alta', 'expedientes']), f"{pct(res.loc['Alta', 'pct_exp'])} de los expedientes · {pesos(res.loc['Alta', 'saldo'])}", 'alta'))
        with c[3]: html(kpi('Efectividad del orden', num(veces) + ' veces', 'más desenlaces favorables en el 10 % mejor priorizado que en un reparto sin priorizar', 'acento'))

        seccion('¿Qué gana el área con el modelo?', 'Mueva la capacidad de gestión del equipo y compare cuánto se cubre siguiendo el orden del modelo frente al orden por fecha de prescripción.')
        izq, der = st.columns([1, 1.25])
        with izq:
            cap = st.slider('Capacidad de gestión (% de la cartera que se alcanza a trabajar)', min_value=5, max_value=60, value=20, step=5, format='%d %%') / 100
            fm, fr = lm.captura(g, 'modelo', cap), lm.captura(g, 'fecha_prescripcion', cap)
            vm, vr = lm.captura(g, 'modelo', cap, 'val'), lm.captura(g, 'fecha_prescripcion', cap, 'val')
            html(f'''<div class="cmp"><div class="t">Desenlaces favorables cubiertos</div>
                 <div class="fila"><div class="g">{pct(fm)}</div><div class="vs">con el modelo, frente a <b>{pct(fr)}</b> por fecha de prescripción</div></div></div>
                 <div class="cmp"><div class="t">Valor pagado cubierto</div>
                 <div class="fila"><div class="g">{pct(vm)}</div><div class="vs">con el modelo, frente a <b>{pct(vr)}</b> por fecha de prescripción</div></div></div>''')
            html(f'<div class="nota">Gestionando el {pct(cap)} de la cartera. Medido sobre expedientes que el modelo no vio al entrenar (validación en cuatro bloques de tiempo); '
                 'son resultados históricos, no una proyección de recaudo.</div>')
        with der:
            fig = lm.figura_ganancia(g, cap); st.pyplot(fig); plt.close(fig)

        seccion('La cartera vigente por prioridad', 'Cómo queda repartida la cartera abierta según el puntaje del modelo.')
        tarjetas_prioridad(res)

        seccion('Los primeros 10 expedientes a gestionar', 'Orden recomendado para el próximo reparto.')
        tabla_orden(car.sort_values('puntaje', ascending=False), 10)

# ============================================================================ un expediente
with tab_uno:
    seccion('Evaluar un expediente', 'Digite los datos del expediente y obtenga su prioridad. Los valores iniciales son un caso ficticio.')
    segmento = st.radio('Tipo de persona', ['juridica', 'natural'], horizontal=True, format_func=lambda s: 'Jurídica' if s == 'juridica' else 'Natural')
    cat = lm.catalogos(PAQ, segmento)
    corte_def = dt.date.fromisoformat(PAQ['segmentos'][segmento]['fecha_corte'][:10])
    with st.form('expediente'):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('**Obligación**')
            valor_multa = st.number_input('Valor de la multa (COP)', min_value=1, value=4_500_000, step=100_000)
            pagado_capital = st.number_input('Pagado a capital (COP)', min_value=0, value=900_000, step=100_000)
            pagado_interes = st.number_input('Pagado a intereses (COP)', min_value=0, value=0, step=50_000)
            anio_origen = st.number_input('Año de origen del expediente', min_value=2005, max_value=corte_def.year, value=2022)
        with c2:
            st.markdown('**Proceso**')
            fecha_corte = st.date_input('Fecha de corte (hoy)', value=corte_def)
            fecha_prescripcion = st.date_input('Fecha de prescripción', value=dt.date(2027, 6, 30))
            fecha_ultimo_acto = st.date_input('Fecha del último acto', value=dt.date(2026, 8, 15))
            ultimo_acto = st.selectbox('Último acto administrativo', cat['actos'], index=cat['actos'].index('acuerdo_pago') if 'acuerdo_pago' in cat['actos'] else 0,
                                       format_func=lambda a: a.replace('_', ' ').capitalize())
            area = st.selectbox('Área generadora', cat['areas'], index=len(cat['areas']) - 1)
        with c3:
            st.markdown('**Sancionado**')
            tipo_id = st.selectbox('Tipo de identificación', cat['tipos_id'])
            otros_dir = st.number_input('Otros expedientes en la misma dirección', min_value=0, value=1)
            otro_dir_pago = st.checkbox('Alguno de ellos registra pagos', value=True)
            otros_cod = st.number_input('Otros códigos del mismo número de expediente', min_value=0, value=0)
            otro_cod_pago = st.checkbox('Alguno de esos códigos registra pagos', value=False)
        enviado = st.form_submit_button('Calcular prioridad', type='primary')

    if enviado:
        dias_presc, dias_acto = (fecha_prescripcion - fecha_corte).days, (fecha_corte - fecha_ultimo_acto).days
        if dias_presc <= 0:
            st.error('La fecha de prescripción debe ser posterior a la fecha de corte: el modelo prioriza expedientes aún vigentes.')
        elif dias_acto < 0:
            st.error('La fecha del último acto no puede ser posterior a la fecha de corte.')
        elif pagado_capital > valor_multa:
            st.error('Lo pagado a capital no puede superar el valor de la multa.')
        else:
            args = dict(dias_desde_ultimo_acto=dias_acto, valor_multa=valor_multa, pagado_capital=pagado_capital, pagado_interes=pagado_interes, anio_origen=anio_origen,
                        area_generadora=area, ultimo_acto=ultimo_acto, tipo_identificacion=tipo_id, otros_exp_misma_direccion=otros_dir,
                        otro_con_pago_misma_direccion=otro_dir_pago, otros_codigos_mismo_expediente=otros_cod, otro_codigo_con_pago=otro_cod_pago)
            r = lm.puntuar(PAQ, lm.construir_fila(PAQ, segmento, dias_a_prescripcion=dias_presc, **args), segmento).iloc[0]
            col = lm.COLOR[r.prioridad]
            ref = PAQ['segmentos'][segmento]['referencia']
            supera = (ref['puntaje'] < r.puntaje).mean()
            izq, der = st.columns([1.25, 1])
            with izq:
                html(f'''<div class="res"><div class="anillo" style="background:conic-gradient({col} {r.puntaje:.1f}%, #E9EDF2 0)"><div>
                     <div class="n" style="color:{col}">{r.puntaje:.0f}</div><div class="d">de 100</div></div></div>
                     <div><span class="badge" style="background:{col}">PRIORIDAD {r.prioridad.upper()}</span>
                     <h3>Probabilidad de desenlace favorable: {pct(r.probabilidad, 1)}</h3>
                     <p>{num(r.veces_tasa_base)} veces la de un expediente promedio de su segmento ({pct(PAQ['segmentos'][segmento]['tasa_base'], 1)}).
                     Supera en puntaje al {pct(supera)} de los {miles(len(ref['puntaje']))} expedientes vigentes.</p>
                     <div class="accion"><b>Acción sugerida:</b> {ACCION[r.prioridad]}</div></div></div>''')
            with der:
                filas = ''
                for nombre, p, w, dato, comparacion in (
                        ('Probabilidad de desenlace favorable', r.pct_probabilidad, PESOS['prob'], pct(r.probabilidad, 1), 'mayor que la del'),
                        ('Valor en juego', r.pct_valor, PESOS['valor'], pesos(r.saldo) + ' de saldo', 'mayor que el del'),
                        ('Urgencia', r.pct_urgencia, PESOS['urgencia'], miles(r.dias_a_prescripcion) + ' días para prescribir', 'más urgente que el')):
                    filas += f'''<div class="f"><div class="c1"><span>{nombre}</span><span>{100 * w * p:.0f} de {100 * w:.0f} pts</span></div>
                              <div class="barra"><div style="width:{p * 100:.1f}%;background:{col}"></div></div>
                              <div class="c2">{dato} · {comparacion} {pct(p)} de la cartera vigente</div></div>'''
                html(f'<div class="comp"><div class="lab" style="font-size:0.74rem;letter-spacing:0.08em;text-transform:uppercase;color:#6B7789;font-weight:600;margin-bottom:14px">¿De dónde sale el puntaje?</div>{filas}</div>')
            seccion('Posición frente a la cartera vigente', f'Distribución del puntaje de los expedientes vigentes ({NOMBRE_SEG[segmento].lower()}); los colores marcan las zonas Baja, Media y Alta.')
            izq, der = st.columns([1.25, 1])
            with izq:
                fig = lm.figura_posicion(PAQ, r, segmento); st.pyplot(fig); plt.close(fig)
            with der:
                html(f'''<div class="cmp"><div class="t">Cómo leerlo</div>
                     <div class="vs" style="margin-top:8px">La línea marca dónde queda este expediente entre los <b>{miles(len(ref['puntaje']))}</b> vigentes de su segmento.
                     Cuanto más a la derecha, antes debería gestionarse.</div></div>
                     <div class="cmp"><div class="t">Zonas de prioridad</div>
                     <div class="vs" style="margin-top:8px"><b style="color:{lm.COLOR['Alta']}">Alta</b> desde {CORTES['alta']} puntos ·
                     <b style="color:{lm.COLOR['Media']}">Media</b> desde {CORTES['media']} · <b style="color:{lm.COLOR['Baja']}">Baja</b> el resto</div></div>''')
            with st.expander('¿Qué pasaría si cambia el tiempo que falta para prescribir?'):
                fig = lm.figura_sensibilidad(PAQ, segmento, dias_presc, args); st.pyplot(fig); plt.close(fig)
                st.caption('Todo lo demás se mantiene igual; la línea punteada es el expediente digitado.')

# ============================================================================ lote
with tab_lote:
    seccion('Priorizar un lote de expedientes', 'Suba un archivo CSV preparado por el notebook de preparación (con las columnas id_expediente y segmento). '
            'Si no sube nada, se usa una muestra de 300 expedientes vigentes que el modelo no usó para entrenar.')
    archivo = st.file_uploader('Archivo CSV de expedientes', type='csv', label_visibility='collapsed')
    try:
        lote = pd.read_csv(archivo if archivo is not None else lm.RUTA_DEMO, dtype={'id_expediente': str})
        pri = lm.puntuar_lote(PAQ, lote)
    except Exception as e:
        st.error(f'No se pudo puntuar el archivo: {e}')
    else:
        rl = lm.resumen_prioridad(pri)
        c = st.columns(3)
        with c[0]: html(kpi('Expedientes puntuados', miles(len(pri)), 'Muestra de demostración' if archivo is None else 'Archivo cargado'))
        with c[1]: html(kpi('Saldo a capital', pesos(pri.saldo.sum()), 'Suma de saldos del lote'))
        with c[2]: html(kpi('Probabilidad media', pct(pri.probabilidad.mean(), 1), 'de desenlace favorable en el lote', 'acento'))
        html('<div style="height:14px"></div>')
        tarjetas_prioridad(rl)
        izq, der = st.columns([1.15, 1])
        with izq:
            seccion('Orden de gestión recomendado', 'De mayor a menor puntaje.')
            tabla_orden(pri)
            st.download_button('Descargar la lista priorizada (CSV)', pri.to_csv().encode('utf-8-sig'), 'priorizacion_lote.csv', 'text/csv', type='primary')
        with der:
            seccion('Probabilidad y valor de cada expediente', 'Arriba a la derecha están los casos más probables y de mayor valor.')
            fig = lm.figura_dispersion(pri); st.pyplot(fig); plt.close(fig)

# ============================================================================ sobre el modelo
with tab_modelo:
    c1, c2 = st.columns([1, 1])
    with c1:
        seccion('Cómo funciona', '')
        st.markdown(f'''
1. **Estima la probabilidad** de que el expediente termine con pago u otra causal favorable en los meses siguientes, a partir de su historial de pagos, el último acto administrativo, el tiempo para prescribir y otros expedientes del mismo sancionado.
2. **Calcula un puntaje de 0 a 100**: {PESOS['prob'] * 100:.0f} % probabilidad, {PESOS['valor'] * 100:.0f} % valor del saldo y {PESOS['urgencia'] * 100:.0f} % urgencia por prescripción.
3. **Asigna la prioridad**: Alta desde {CORTES['alta']} puntos, Media desde {CORTES['media']} y Baja el resto.

Los pesos y los cortes son parámetros que define el área de cobro.''')
    with c2:
        seccion('Qué se comprobó en la validación', '')
        mt = {s: PAQ['segmentos'][s]['metricas_prueba'] for s in PAQ['segmentos']}
        items = [(True, 'Ordena mejor que la fecha de prescripción', 'La mejora en el 10 % de mayor prioridad supera el 20 % exigido y es estadísticamente significativa.'),
                 (True, 'La probabilidad es confiable', f"Error de calibración de {num(mt['natural']['ECE'], 3)} y {num(mt['juridica']['ECE'], 3)}, frente a un límite de 0,10."),
                 (False, 'No identifica cada caso individual', f"Gestionando el 10 % de mayor prioridad encuentra el {pct(mt['natural']['Recall'])} (naturales) y el {pct(mt['juridica']['Recall'])} (jurídicas) de los desenlaces favorables; la meta era 70 %. Sirve para ordenar, no para decidir caso por caso.")]
        html(''.join(f'<div class="check"><div class="i" style="background:{"#0E9F8E" if ok else "#C8102E"}">{"✓" if ok else "!"}</div><div class="tx"><b>{t}</b><br>{d}</div></div>' for ok, t, d in items))
    seccion('Ficha técnica', 'Métricas sobre el conjunto de prueba (los expedientes más recientes, que no participaron en el entrenamiento).')
    ficha = pd.DataFrame({NOMBRE_SEG[s]: {'Clasificador': d['familia'], 'Expedientes de entrenamiento': miles(d['n_entrenamiento']), 'Variables': str(len(d['variables'])),
                                          'Tasa base de desenlace favorable': pct(d['tasa_base'], 1), 'ROC-AUC': num(d['metricas_prueba']['ROC-AUC'], 3), 'PR-AUC': num(d['metricas_prueba']['PR-AUC'], 3),
                                          'Precision / Recall / F1 (10 % de mayor prioridad)': f"{num(d['metricas_prueba']['Precision'], 2)} / {num(d['metricas_prueba']['Recall'], 2)} / {num(d['metricas_prueba']['F1'], 2)}",
                                          'Lift en el decil superior': num(d['metricas_prueba']['Lift decil 1'], 2), 'Error de calibración': num(d['metricas_prueba']['ECE'], 3),
                                          'Cartera vigente de referencia': miles(len(d['referencia']['prob'])) + ' expedientes'} for s, d in PAQ['segmentos'].items()})
    st.table(ficha)
    html('<div class="nota">Limitaciones principales: el desenlace se observó durante 8 meses; hay cerca de cien casos favorables por segmento en la prueba; no se cuenta con el historial de gestiones. '
         'El efecto sobre el recaudo y la prescripción solo puede confirmarse con un piloto de reparto. El modelo debe reentrenarse con cada nuevo corte de información.</div>')
