"""Aplicación Streamlit: priorización de la cartera de cobro coactivo del FFDS.

Ejecutar desde la carpeta del proyecto:   streamlit run app.py
"""
import datetime as dt
import pandas as pd
import sklearn
import streamlit as st
import matplotlib.pyplot as plt
import logica_modelo as lm

st.set_page_config(page_title='Priorización de cobro coactivo – FFDS', page_icon='⚖️', layout='wide')


@st.cache_resource
def paquete_modelo():
    return lm.cargar_paquete()


try:
    PAQ = paquete_modelo()
except Exception as e:                                   # archivo ausente o versión incompatible de scikit-learn
    st.error('No se pudo cargar el modelo (modelo/modelo_priorizacion_ffds.joblib). Ejecute el Notebook 1 en este mismo entorno para generarlo.')
    st.exception(e)
    st.stop()

PESOS, CORTES = PAQ['pesos_puntaje'], PAQ['cortes']

# ---------------------------------------------------------------- barra lateral
with st.sidebar:
    st.header('Modelo')
    st.write(f"**{PAQ['nombre']}**")
    st.caption(f"Versión {PAQ['version']} · entrenado el {PAQ['fecha_entrenamiento']}")
    st.write(f"Puntaje = 100 × ({PESOS['prob']} × probabilidad + {PESOS['valor']} × valor + {PESOS['urgencia']} × urgencia)")
    st.write(f"Prioridad **Alta** ≥ {CORTES['alta']} · **Media** ≥ {CORTES['media']} · **Baja** el resto")
    if PAQ['versiones']['scikit-learn'] != sklearn.__version__:
        st.warning(f"El modelo se guardó con scikit-learn {PAQ['versiones']['scikit-learn']} y aquí hay {sklearn.__version__}. Si algo falla, regenere el modelo con el Notebook 1.")
    st.divider()
    st.caption('El modelo recomienda un orden de gestión; la decisión jurídica sobre cada expediente sigue siendo del abogado sustanciador.')

st.title('Priorización predictiva de la cartera de cobro coactivo')
st.caption('Fondo Financiero Distrital de Salud · Proyecto de Machine Learning Probabilístico · Konrad Lorenz')
tab_uno, tab_lote, tab_ficha = st.tabs(['Un expediente', 'Lote de expedientes', 'Ficha del modelo'])

# ---------------------------------------------------------------- un expediente
with tab_uno:
    segmento = st.radio('Tipo de persona', ['juridica', 'natural'], horizontal=True, format_func=lambda s: 'Jurídica' if s == 'juridica' else 'Natural')
    cat = lm.catalogos(PAQ, segmento)
    corte_def = dt.date.fromisoformat(PAQ['segmentos'][segmento]['fecha_corte'][:10])
    with st.form('expediente'):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader('Obligación')
            valor_multa = st.number_input('Valor de la multa (COP)', min_value=1, value=4_500_000, step=100_000)
            pagado_capital = st.number_input('Pagado a capital (COP)', min_value=0, value=900_000, step=100_000)
            pagado_interes = st.number_input('Pagado a intereses (COP)', min_value=0, value=0, step=50_000)
            anio_origen = st.number_input('Año de origen del expediente', min_value=2005, max_value=corte_def.year, value=2022)
        with c2:
            st.subheader('Proceso')
            fecha_corte = st.date_input('Fecha de corte (hoy)', value=corte_def)
            fecha_prescripcion = st.date_input('Fecha de prescripción', value=dt.date(2027, 6, 30))
            fecha_ultimo_acto = st.date_input('Fecha del último acto', value=dt.date(2026, 8, 15))
            ultimo_acto = st.selectbox('Último acto administrativo', cat['actos'], index=cat['actos'].index('acuerdo_pago') if 'acuerdo_pago' in cat['actos'] else 0)
            area = st.selectbox('Área generadora', cat['areas'], index=len(cat['areas']) - 1)
        with c3:
            st.subheader('Sancionado')
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
            res = lm.puntuar(PAQ, lm.construir_fila(PAQ, segmento, dias_a_prescripcion=dias_presc, **args), segmento).iloc[0]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric('Prioridad', res.prioridad.upper())
            m2.metric('Puntaje de priorización', f'{res.puntaje:.0f} / 100')
            m3.metric('Probabilidad de desenlace favorable', f'{res.probabilidad * 100:.1f} %', f'{res.veces_tasa_base:.1f} veces la tasa base', delta_color='off')
            m4.metric('Saldo a capital', f'${res.saldo:,.0f}', f'{res.dias_a_prescripcion:,.0f} días para prescribir', delta_color='off')
            fig = lm.figura_tarjeta(PAQ, res, segmento); st.pyplot(fig); plt.close(fig)
            with st.expander('¿Qué pasaría si cambia el tiempo que falta para prescribir?'):
                fig = lm.figura_sensibilidad(PAQ, segmento, dias_presc, args); st.pyplot(fig); plt.close(fig)
                st.caption('Todo lo demás se mantiene igual; la línea punteada es el expediente digitado.')
    else:
        st.info('Complete los datos y pulse **Calcular prioridad**. Los valores iniciales son un caso ficticio.')

# ---------------------------------------------------------------- lote
with tab_lote:
    st.write('Suba un archivo CSV **preparado** (salida del notebook de preparación, con las columnas `id_expediente` y `segmento`). Si no sube nada, se usa la muestra de 300 expedientes vigentes incluida con el modelo.')
    archivo = st.file_uploader('Archivo CSV de expedientes', type='csv')
    try:
        lote = pd.read_csv(archivo if archivo is not None else lm.RUTA_DEMO, dtype={'id_expediente': str})
        pri = lm.puntuar_lote(PAQ, lote)
    except Exception as e:
        st.error(f'No se pudo puntuar el archivo: {e}')
    else:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric('Expedientes', f'{len(pri):,}')
        k2.metric('Prioridad Alta', f"{(pri.prioridad == 'Alta').sum():,}")
        k3.metric('Prioridad Media', f"{(pri.prioridad == 'Media').sum():,}")
        k4.metric('Saldo en prioridad Alta', f"${pri.loc[pri.prioridad == 'Alta', 'saldo'].sum() / 1e6:,.1f} M")
        fig = lm.figura_lote(pri); st.pyplot(fig); plt.close(fig)
        st.subheader('Orden de gestión recomendado')
        vista = pri[['id_expediente', 'segmento', 'prioridad', 'puntaje', 'probabilidad', 'saldo', 'dias_a_prescripcion']].copy()
        vista['puntaje'] = vista['puntaje'].round(0); vista['probabilidad'] = (vista['probabilidad'] * 100).round(1)
        st.dataframe(vista.rename(columns={'probabilidad': 'probabilidad (%)', 'saldo': 'saldo (COP)', 'dias_a_prescripcion': 'días a prescripción'}), height=380)
        st.download_button('Descargar la lista priorizada (CSV)', pri.to_csv().encode('utf-8-sig'), 'priorizacion_lote.csv', 'text/csv')

# ---------------------------------------------------------------- ficha
with tab_ficha:
    st.write(f"**Qué predice:** {PAQ['objetivo']}.")
    ficha = pd.DataFrame({('Persona ' + s): {'Clasificador': d['familia'], 'Hiperparámetros': str(d['hiperparametros']), 'Expedientes de entrenamiento': f"{d['n_entrenamiento']:,}",
                                           'Variables': str(len(d['variables'])), 'Tasa base de desenlace favorable': f"{d['tasa_base'] * 100:.1f} %",
                                           'ROC-AUC (prueba)': f"{d['metricas_prueba']['ROC-AUC']:.3f}", 'PR-AUC (prueba)': f"{d['metricas_prueba']['PR-AUC']:.3f}",
                                           'Precision / Recall / F1 (prueba)': f"{d['metricas_prueba']['Precision']:.2f} / {d['metricas_prueba']['Recall']:.2f} / {d['metricas_prueba']['F1']:.2f}",
                                           'Lift en el decil superior (prueba)': f"{d['metricas_prueba']['Lift decil 1']:.2f}", 'Error de calibración (prueba)': f"{d['metricas_prueba']['ECE']:.3f}",
                                           'Cartera vigente de referencia': f"{len(d['referencia']['prob']):,} expedientes al {d['fecha_corte'][:10]}"} for s, d in PAQ['segmentos'].items()})
    st.table(ficha)
    st.subheader('Cómo leer el resultado')
    st.markdown('- **Probabilidad:** estimación calibrada de que el expediente termine con pago u otra causal favorable en los meses siguientes.\n'
                '- **Puntaje (0–100):** combina probabilidad, valor en juego y urgencia, cada uno como posición frente a la cartera vigente. Define el orden del reparto.\n'
                '- **Límites:** el modelo ordena mejor que la regla de fecha de prescripción, pero no acierta caso por caso (recall de 0,25 con el umbral de clasificación). '
                'Debe reentrenarse con cada nuevo corte de información.')
