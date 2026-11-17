# ML-Prescripciones
Modelo de clasificación para priorización de expedientes en gestión de cobro
================================================================================
PRIORIZACIÓN PREDICTIVA DE LA CARTERA DE COBRO COACTIVO DEL FFDS
Proyecto grupal final - Parte 2 (CRISP-DM fases 4, 5 y 6)
Machine Learning Probabilístico - Especialización en Analítica Estratégica de Datos
Fundación Universitaria Konrad Lorenz - Profesor Carlos Amortegui
Integrantes: Diana Murillo Ferro, Edison Riaño Umaña, Carolina Sáchica Patiño
================================================================================

1. QUÉ HAY EN ESTA CARPETA
--------------------------------------------------------------------------------
Proyecto_Parte2/
  README.txt                                  Este archivo
  Notebook_1_Modelamiento_Evaluacion.ipynb    Fases 4 y 5: entrena, compara, evalúa y GUARDA el modelo
  Notebook_2_Despliegue_Demo.ipynb            Fase 6: CARGA el modelo y puntúa expedientes nuevos
  datos/                                      Datos preparados (salida de las fases 1 a 3), sin datos personales
      X_train_<segmento>.csv, X_test_<segmento>.csv   Expedientes etiquetados (corte 28-02-2026)
      X_actual_<segmento>.csv                         Expedientes vigentes por priorizar (corte 31-10-2026)
      metadatos_preparacion_<segmento>.json           Parámetros de la preparación
      Datos_Preparados_sin_PII.csv                    Base preparada (se usa para el catálogo de actos)
  modelo/
      modelo_priorizacion_ffds.joblib          Modelo serializado (lo genera el Notebook 1)
      demo_expedientes_nuevos.csv              300 expedientes vigentes para la demostración
  resultados/                                  Tablas (.csv) y figuras (.png) que generan los notebooks
  preparacion/
      EDA_Prescripciones_VF_ajustado.ipynb     Notebook de las fases 1 a 3 (referencia)
  app.py                                      Aplicación Streamlit para probar el modelo en vivo
  logica_modelo.py                            Funciones de puntuación que usa la aplicación
  requirements.txt                            Librerías de la aplicación

<segmento> = natural | juridica (personas naturales y jurídicas se modelan por separado).

2. REQUISITOS
--------------------------------------------------------------------------------
Python 3.10 o superior con: numpy, pandas, matplotlib, scikit-learn, joblib.
Google Colab ya los trae instalados; no hay que instalar nada.
Versiones con las que se generó esta entrega:
  Python 3.13 | scikit-learn 1.9.1 | numpy 2.5 | pandas 3.0 | joblib 1.6 | matplotlib 3.11
No se necesita GPU. Tiempos aproximados con 2 núcleos: Notebook 1 = 2 a 4 minutos;
Notebook 2 = menos de 30 segundos.

3. CÓMO REPRODUCIR EN GOOGLE COLAB (recomendado)
--------------------------------------------------------------------------------
Paso 1. Suba la carpeta completa a Google Drive en esta ruta exacta:
          Mi unidad/Konrad/EDA_FFDS/Proyecto_Parte2
        (si usa otra ruta, cámbiela en la primera celda de código de cada
         notebook, en la lista _candidatas).
Paso 2. Abra Notebook_1_Modelamiento_Evaluacion.ipynb con Google Colab.
Paso 3. Menú "Entorno de ejecución" -> "Ejecutar todas". Autorice el acceso a Drive
        cuando lo pida.
Paso 4. Al terminar, la última celda debe mostrar:
          "El modelo serializado se carga y reproduce las predicciones."
        y existirá el archivo modelo/modelo_priorizacion_ffds.joblib.
Paso 5. Abra Notebook_2_Despliegue_Demo.ipynb y ejecute todas las celdas.
Paso 6. Para probar otro expediente, cambie los campos del formulario del paso 3
        del Notebook 2 (aparece a la derecha de la celda) y ejecute esa celda y
        la siguiente.

IMPORTANTE: ejecute siempre el Notebook 1 antes que el Notebook 2 en el mismo
entorno. Un archivo .joblib solo se puede cargar con una versión de scikit-learn
igual o muy cercana a la que lo guardó; el Notebook 1 lo regenera en el entorno
en que usted está trabajando. El Notebook 2 avisa si las versiones no coinciden.

4. CÓMO REPRODUCIR EN UN COMPUTADOR (Jupyter o VS Code)
--------------------------------------------------------------------------------
Paso 1. Instale las librerías:
          pip install numpy pandas matplotlib scikit-learn joblib jupyter
Paso 2. Abra una terminal dentro de la carpeta Proyecto_Parte2 y ejecute:
          jupyter notebook
Paso 3. Ejecute Notebook_1 completo y luego Notebook_2 completo.
        Los notebooks detectan la carpeta actual si contiene datos/ y modelo/.
        También puede indicar la carpeta con la variable de entorno
        RUTA_PROYECTO_FFDS.

5. CÓMO USAR EL MODELO CON DATOS NUEVOS (Notebook 2)
--------------------------------------------------------------------------------
Opción A - Un expediente: llene el formulario (tipo de persona, valor de la
  multa, pagos, fecha de prescripción, fecha y tipo del último acto, área
  generadora, año de origen y otros expedientes del mismo sancionado). El
  notebook devuelve la probabilidad de desenlace favorable, el puntaje de
  priorización (0-100) y la prioridad Alta / Media / Baja, con una tarjeta visual.
Opción B - Un lote: deje ARCHIVO_LOTE apuntando a modelo/demo_expedientes_nuevos.csv
  o cámbielo por un X_actual_<segmento>.csv (agregando la columna "segmento").
  El resultado se guarda en resultados/demo_priorizacion_lote.csv.

Uso desde código propio (mínimo):
    import joblib
    paquete = joblib.load('modelo/modelo_priorizacion_ffds.joblib')
    seg = paquete['segmentos']['juridica']        # o 'natural'
    # X: DataFrame con las columnas seg['variables'], en ese orden
    # la probabilidad calibrada se obtiene con la función puntuar() del Notebook 2

6. QUÉ PRODUCE CADA NOTEBOOK
--------------------------------------------------------------------------------
Notebook 1 -> resultados/
  hiperparametros_probados.csv        Todas las configuraciones evaluadas
  tabla_comparativa_modelos.csv       Accuracy, Precision, Recall, F1, AUC y matriz de confusión
  matrices_confusion_prueba.png       Matrices de confusión por modelo y segmento
  metricas_negocio_escenarios.csv     Comparación de los tres escenarios de priorización
  bootstrap_vs_regla_trivial.csv      Significancia frente a la regla trivial
  sensibilidad_pesos.csv              Efecto de los pesos del puntaje
  matriz_decision.csv                 Elección del escenario final
  modelo_final_prueba.png             Matriz de confusión, curva de ganancia y calibración
  deciles_prueba.csv                  Tabla por deciles en el conjunto de prueba
  importancia_variables.csv / .png    Variables más influyentes
  respuestas_negocio.csv              Cifras que responden las preguntas de la fase 1
  priorizacion_expedientes_vigentes.csv   Cartera vigente priorizada (27.627 expedientes)
Notebook 1 -> modelo/
  modelo_priorizacion_ffds.joblib y demo_expedientes_nuevos.csv
Notebook 2 -> resultados/
  demo_tarjeta_expediente.png, demo_tablero_lote.png, demo_priorizacion_lote.csv

7. REPRODUCIBILIDAD Y PARÁMETROS
--------------------------------------------------------------------------------
* Semilla fija (SEMILLA = 42): los resultados se repiten entre ejecuciones con las
  mismas versiones de librerías. Con otras versiones pueden cambiar en el tercer
  decimal.
* Parámetros de negocio (primera celda de código del Notebook 1): pesos del puntaje
  PESOS_PUNTAJE = 0,7 probabilidad / 0,2 valor / 0,1 urgencia; cortes de prioridad
  80 (Alta) y 50 (Media). Si se cambian, vuelva a ejecutar ambos notebooks.
* Para regenerar la carpeta datos/ desde los archivos originales de Excel (no se
  incluyen porque contienen datos personales) ejecute
  preparacion/EDA_Prescripciones_VF_ajustado.ipynb con
  CONJUNTO_VARIABLES = 'mejorado' y TRATAMIENTO_CENSURA = 'como_cero', y copie a
  datos/ los archivos X_*.csv, metadatos_*.json y Datos_Preparados_sin_PII.csv.

8. PROBLEMAS FRECUENTES
--------------------------------------------------------------------------------
* "StopIteration" en la primera celda: el notebook no encontró la carpeta del
  proyecto. Revise la ruta de Drive (paso 1 de la sección 3).
* Error al cargar el .joblib o aviso de versión distinta de scikit-learn: ejecute
  de nuevo el Notebook 1 en ese mismo entorno.
* "Exporte la preparación con TRATAMIENTO_CENSURA = como_cero": los datos de
  datos/ no son los de esta entrega; vuelva a copiarlos.

9. APLICACIÓN STREAMLIT
--------------------------------------------------------------------------------
La aplicación usa el mismo archivo modelo/modelo_priorizacion_ffds.joblib y las
mismas funciones del Notebook 2. Tiene tres pestañas: "Un expediente" (formulario
y tarjeta de resultado), "Lote de expedientes" (carga de un CSV preparado y
descarga de la lista priorizada) y "Ficha del modelo".

En un computador (opción recomendada para la sustentación):
Paso 1. Abra una terminal dentro de la carpeta Proyecto_Parte2.
Paso 2. Instale las librerías:      pip install -r requirements.txt
Paso 3. Inicie la aplicación:       streamlit run app.py
Paso 4. Se abre el navegador en http://localhost:8501. Para cerrarla: Ctrl + C.

Si aparece un error al cargar el modelo, la versión de scikit-learn instalada no
coincide con la del archivo .joblib: ejecute el Notebook 1 en ese mismo entorno
(regenera el modelo) y, si usa requirements.txt, ponga allí esa versión.

En internet (Streamlit Community Cloud, gratuito):
Paso 1. Suba a un repositorio de GitHub: app.py, logica_modelo.py,
        requirements.txt y la carpeta modelo/ (el .joblib pesa 1,3 MB y la
        muestra de demostración no tiene datos personales).
Paso 2. En share.streamlit.io elija "Create app", seleccione el repositorio y
        app.py como archivo principal. En "Advanced settings" escoja una versión
        de Python compatible con el scikit-learn de requirements.txt.
Paso 3. La aplicación queda con una dirección pública que puede abrir desde
        cualquier equipo. No suba la carpeta datos/ si el repositorio es público.

10. ALCANCE Y ADVERTENCIA DE USO
--------------------------------------------------------------------------------
El modelo recomienda un orden de gestión; no decide sobre el fondo de ningún
proceso. Los archivos no contienen datos personales (sin nombres, identificaciones
ni direcciones). Limitaciones y mejoras propuestas: sección 5.5 del Notebook 1.
