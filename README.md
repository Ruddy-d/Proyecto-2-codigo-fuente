**#Equipo: Stalin Perez, Angel Perez y Felix Terrero**

🏥 Clasificación Avanzada de Cáncer de Mama

Proyecto de Machine Learning con comparación de modelos y análisis visual

**📌 1. Descripción General**

Este proyecto implementa un sistema de clasificación para el dataset Breast Cancer Wisconsin utilizando varios modelos de Machine Learning.

🎯 Objetivo: comparar distintos algoritmos y determinar cuál ofrece mejor rendimiento para apoyar el diagnóstico temprano del cáncer de mama.

**⚙️ 2. ¿Qué hace el proyecto?**

El script realiza los siguientes pasos:

1️⃣ Carga el dataset de cáncer de mama
2️⃣ Explora la información básica del dataset
3️⃣ Divide los datos en entrenamiento y prueba (80% / 20%)
4️⃣ Estandariza las características con StandardScaler
5️⃣ Entrena varios modelos de Machine Learning
6️⃣ Evalúa cada modelo con distintas métricas
7️⃣ Genera visualizaciones comparativas
8️⃣ Realiza la predicción de un caso real
9️⃣ Muestra un ranking final de modelos

**🧠 3. Modelos Utilizados**

Se entrenan y comparan los siguientes modelos:

🔹 K-Nearest Neighbors (KNN)

🌲 Random Forest

📐 Support Vector Machine (SVM)

📊 Regresión Logística

🚀 Gradient Boosting

Todos los modelos utilizan datos escalados para asegurar una comparación justa.

**📊 4. Métricas de Evaluación**

Cada modelo se evalúa utilizando:

✅ Accuracy

🔁 Validación cruzada (5-Fold)

📈 AUC-ROC

📋 Reporte de clasificación

🧮 Matriz de confusión

**📈 5. Visualizaciones Generadas**

El proyecto genera gráficos para facilitar la interpretación:

📊 Comparación de precisión entre modelos

📈 Curvas ROC

🧮 Matriz de confusión del mejor modelo

🔁 Validación cruzada con desviación estándar

🔍 Importancia de características (si el modelo lo permite)

🥧 Distribución de clases del dataset

📉 Histogramas de características relevantes

🗂️ Archivos generados:

analisis_cancer_completo.png

distribucion_caracteristicas.png

**🔄 6. Flujo del Proyecto**

1️⃣ Carga del dataset
2️⃣ Exploración de datos
3️⃣ División Train/Test
4️⃣ Escalado de datos
5️⃣ Entrenamiento de modelos
6️⃣ Evaluación de resultados
7️⃣ Visualización gráfica
8️⃣ Predicción de un nuevo caso
9️⃣ Ranking final

**🏆 7. Resultados Destacados**

🥇 Los modelos ensemble (Random Forest y Gradient Boosting) presentan mejor rendimiento

🎯 Precisión aproximada entre 95% y 99%

🔑 Identificación de las características más influyentes

**📁 8. Requisitos del Proyecto**

🐍 Python 3.8 o superior

📦 Bibliotecas necesarias:

numpy

pandas

matplotlib

seaborn

scikit-learn

📥 Instalación recomendada:

pip install numpy pandas matplotlib seaborn scikit-learn

**▶️ 9. Ejecución del Proyecto**

Ejecuta el archivo principal:

python nombre_del_archivo.py


📌 Esto generará:

Resultados completos en consola

Gráficos en formato PNG

Comparación de modelos

Resumen del mejor modelo

**📝 10. Notas Finales**

Este proyecto es útil para:

📘 Aprender comparación de modelos de Machine Learning

🧠 Comprender métricas clave en clasificación médica

🛠️ Practicar análisis visual de datos

❤️ Aplicar Machine Learning a un problema real de salud

