#Equipo: Stalin Perez, Angel Perez y Felix Terrero

Clasificación Avanzada de Cáncer de Mama

Proyecto de Machine Learning con múltiples modelos, análisis visual y evaluación completa

📌 Descripción General

Este proyecto implementa un sistema de clasificación para el dataset Breast Cancer Wisconsin usando varios modelos de Machine Learning.
Incluye:

Preparación y estandarización de datos

Entrenamiento de varios clasificadores

Validación cruzada

Curvas ROC

Matriz de confusión

Importancia de características

Análisis exploratorio de datos (EDA)

Comparación de modelos

Predicción de un caso real del dataset

Ajuste de hiperparámetros con GridSearchCV

El objetivo es identificar el modelo con mejor rendimiento para apoyar el diagnóstico temprano del cáncer de mama.

🧠 Modelos Utilizados

KNN

Random Forest

SVM

Logistic Regression

Gradient Boosting

Cada modelo se entrena con datos escalados y se evalúa en métricas como:

Accuracy

Validación cruzada (CV)

AUC-ROC

Reporte de clasificación

📊 Visualizaciones Generadas

El script produce varios gráficos en alta calidad:

Comparación de precisión entre modelos

Curvas ROC

Matriz de confusión del mejor modelo

Validación cruzada con desviación estándar

Importancia de características (si el modelo lo permite)

Distribución de clases

Archivos generados:

analisis_cancer_completo.png

distribucion_caracteristicas.png

🔍 Flujo del Proyecto

Carga y exploración del dataset

Separación Train/Test (80/20)

Estandarización con StandardScaler

Entrenamiento de los modelos

Evaluación y comparación

Visualización gráfica

Predicción de un nuevo caso

Ajuste de hiperparámetros para SVM

Reporte final y ranking de modelos

🏆 Resultados Destacados

Modelos tipo ensemble (Random Forest, Gradient Boosting) suelen rendir mejor.

El sistema alcanza una precisión aproximada del 95–99%, dependiendo del modelo.

Se identifican las características más relevantes en la clasificación.

⚙️ Ajuste de Hiperparámetros (GridSearchCV)

Se realiza una búsqueda en rejilla sobre:

C: [0.1, 1, 10, 100]
gamma: ['scale', 'auto', 0.001, 0.01, 0.1]
kernel: ['rbf', 'linear']


El mejor modelo ajustado se evalúa nuevamente en test.

📁 Requisitos

Python 3.8+

Bibliotecas:

numpy
pandas
matplotlib
seaborn
scikit-learn


Instalación recomendada:

pip install numpy pandas matplotlib seaborn scikit-learn

▶️ Ejecución

Solo ejecuta el archivo principal:

python nombre_del_archivo.py


Esto generará:

Resultados completos en consola

Gráficos PNG

Comparaciones de modelos

Rendimiento del modelo optimizado

📝 Notas Finales

Este proyecto es útil para:

Aprender evaluación comparativa de modelos

Entender métricas clave en clasificación médica

Practicar ajuste de hiperparámetros

Realizar análisis visual y explicativo


