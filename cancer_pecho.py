from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                            roc_curve, auc, roc_auc_score)
####################################################################################
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print(" " * 20 + "🏥 CLASIFICACIÓN AVANZADA DE CÁNCER DE PECHO 🏥")
print("=" * 80)

datos = load_breast_cancer()
X = datos.data
y = datos.target
############################################################################
df = pd.DataFrame(X, columns=datos.feature_names)
df['target'] = y

print(f"\n📊 INFORMACIÓN DEL DATASET")
print(f"   Total de muestras: {X.shape[0]}")
print(f"   Características por muestra: {X.shape[1]}")
print(f"   Clases: {list(datos.target_names)}")

casos_malignos = np.sum(y == 0)
casos_benignos = np.sum(y == 1)
print(f"   • Casos malignos: {casos_malignos} ({casos_malignos/len(y)*100:.1f}%)")
print(f"   • Casos benignos: {casos_benignos} ({casos_benignos/len(y)*100:.1f}%)")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
############################################################################27/11/2023

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

modelos = {
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42, C=1.0, gamma='scale'),
    'Logistic Regression': LogisticRegression(max_iter=10000, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42, learning_rate=0.1)
}

resultados = {}
predicciones = {}
probabilidades = {}

print(f"\n{'=' * 80}")
print("🔬 ENTRENAMIENTO Y EVALUACIÓN DE MODELOS")
print(f"{'=' * 80}\n")

for nombre, modelo in modelos.items():
    print(f"Entrenando {nombre}...", end=" ")
    
    modelo.fit(X_train_scaled, y_train)
    
    y_pred = modelo.predict(X_test_scaled)
    y_proba = modelo.predict_proba(X_test_scaled)[:, 1]
    
    predicciones[nombre] = y_pred
    probabilidades[nombre] = y_proba
    
    accuracy = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(modelo, X_train_scaled, y_train, cv=5)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    resultados[nombre] = {
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'roc_auc': roc_auc
    }
    
    print(f"✓ Precisión: {accuracy:.2%} | CV: {cv_scores.mean():.2%} (±{cv_scores.std():.2%}) | AUC: {roc_auc:.3f}")

print(f"\n{'=' * 80}")
print("📈 RESULTADOS DETALLADOS POR MODELO")
print(f"{'=' * 80}\n")

mejor_modelo_nombre = max(resultados, key=lambda x: resultados[x]['accuracy'])

for nombre in modelos.keys():
    print(f"\n{'─' * 80}")
    print(f"📊 {nombre}")
    print(f"{'─' * 80}")
    print(classification_report(y_test, predicciones[nombre], target_names=datos.target_names))

fig = plt.figure(figsize=(20, 12))
fig.suptitle('🏥 ANÁLISIS COMPLETO DE CLASIFICACIÓN DE CÁNCER DE PECHO 🏥', fontsize=20, fontweight='bold', y=0.995)

ax1 = plt.subplot(2, 3, 1)
nombres_modelos = list(resultados.keys())
accuracies = [resultados[m]['accuracy'] for m in nombres_modelos]
bars = ax1.barh(nombres_modelos, accuracies, color=sns.color_palette("husl", len(nombres_modelos)))
ax1.set_xlabel('Precisión', fontsize=12, fontweight='bold')
ax1.set_title('Comparación de Precisión de Modelos', fontsize=14, fontweight='bold')
ax1.set_xlim([0.9, 1.0])
for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    ax1.text(acc, i, f' {acc:.2%}', va='center', fontweight='bold')
ax1.axvline(x=0.95, color='red', linestyle='--', alpha=0.5, label='95%')
ax1.legend()

ax2 = plt.subplot(2, 3, 2)
for nombre in modelos.keys():
    fpr, tpr, _ = roc_curve(y_test, probabilidades[nombre])
    auc_score = auc(fpr, tpr)
    ax2.plot(fpr, tpr, label=f'{nombre} (AUC={auc_score:.3f})', linewidth=2)
ax2.plot([0, 1], [0, 1], 'k--', label='Azar', linewidth=1)
ax2.set_xlabel('Tasa de Falsos Positivos', fontsize=12, fontweight='bold')
ax2.set_ylabel('Tasa de Verdaderos Positivos', fontsize=12, fontweight='bold')
ax2.set_title('Curvas ROC de Todos los Modelos', fontsize=14, fontweight='bold')
ax2.legend(loc='lower right', fontsize=9)
ax2.grid(True, alpha=0.3)

ax3 = plt.subplot(2, 3, 3)
cm = confusion_matrix(y_test, predicciones[mejor_modelo_nombre])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=datos.target_names,
            yticklabels=datos.target_names, cbar_kws={'label': 'Frecuencia'}, ax=ax3)
ax3.set_ylabel('Etiqueta Real', fontsize=12, fontweight='bold')
ax3.set_xlabel('Etiqueta Predicha', fontsize=12, fontweight='bold')
ax3.set_title(f'Matriz de Confusión - {mejor_modelo_nombre}', fontsize=14, fontweight='bold')

ax4 = plt.subplot(2, 3, 4)
cv_means = [resultados[m]['cv_mean'] for m in nombres_modelos]
cv_stds = [resultados[m]['cv_std'] for m in nombres_modelos]
ax4.barh(nombres_modelos, cv_means, xerr=cv_stds, color=sns.color_palette("husl", len(nombres_modelos)),
         capsize=5, alpha=0.8)
ax4.set_xlabel('Precisión (Validación Cruzada)', fontsize=12, fontweight='bold')
ax4.set_title('Validación Cruzada (5-Fold) con Desviación Estándar', fontsize=14, fontweight='bold')
ax4.set_xlim([0.9, 1.0])
for i, (mean, std) in enumerate(zip(cv_means, cv_stds)):
    ax4.text(mean, i, f' {mean:.2%}±{std:.2%}', va='center', fontsize=9, fontweight='bold')

mejor_modelo = modelos[mejor_modelo_nombre]
if hasattr(mejor_modelo, 'feature_importances_'):
    ax5 = plt.subplot(2, 3, 5)
    importancias = mejor_modelo.feature_importances_
    indices = np.argsort(importancias)[-10:]
    ax5.barh(range(10), importancias[indices], color=sns.color_palette("rocket", 10))
    ax5.set_yticks(range(10))
    ax5.set_yticklabels([datos.feature_names[i] for i in indices], fontsize=9)
    ax5.set_xlabel('Importancia', fontsize=12, fontweight='bold')
    ax5.set_title(f'Top 10 Características - {mejor_modelo_nombre}', fontsize=14, fontweight='bold')
else:
    ax5 = plt.subplot(2, 3, 5)
    ax5.text(0.5, 0.5, f'{mejor_modelo_nombre}\nno proporciona\nimportancia de características',
             ha='center', va='center', fontsize=12, transform=ax5.transAxes)
    ax5.set_title(f'Importancia de Características - {mejor_modelo_nombre}', fontsize=14, fontweight='bold')
    ax5.axis('off')

ax6 = plt.subplot(2, 3, 6)
clases_counts = [casos_malignos, casos_benignos]
colores = ['#FF6B6B', '#4ECDC4']
wedges, texts, autotexts = ax6.pie(clases_counts, labels=datos.target_names, autopct='%1.1f%%',
                                     colors=colores, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
ax6.set_title('Distribución de Clases en el Dataset', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('analisis_cancer_completo.png', dpi=300, bbox_inches='tight')
print(f"\n💾 Gráfico guardado como 'analisis_cancer_completo.png'")
plt.show()

fig2, axes = plt.subplots(2, 2, figsize=(16, 12))
fig2.suptitle('📊 ANÁLISIS EXPLORATORIO DE DATOS', fontsize=18, fontweight='bold')

caracteristicas_importantes = ['mean radius', 'mean texture', 'mean perimeter', 'mean area']

for idx, caracteristica in enumerate(caracteristicas_importantes):
    ax = axes[idx // 2, idx % 2]
    for clase in [0, 1]:
        datos_clase = df[df['target'] == clase][caracteristica]
        ax.hist(datos_clase, bins=30, alpha=0.6, label=datos.target_names[clase], edgecolor='black')
    ax.set_xlabel(caracteristica.title(), fontsize=11, fontweight='bold')
    ax.set_ylabel('Frecuencia', fontsize=11, fontweight='bold')
    ax.set_title(f'Distribución: {caracteristica.title()}', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('distribucion_caracteristicas.png', dpi=300, bbox_inches='tight')
print(f"💾 Gráfico guardado como 'distribucion_caracteristicas.png'")
plt.show()

print(f"\n{'=' * 80}")
print("🎯 PREDICCIÓN CON CASO NUEVO")
print(f"{'=' * 80}\n")

caso_ejemplo = X_test[0].reshape(1, -1)
caso_escalado = scaler.transform(caso_ejemplo)
etiqueta_real = datos.target_names[y_test[0]]

print(f"Características del caso (primeras 5):")
for i in range(5):
    print(f"  • {datos.feature_names[i]}: {caso_ejemplo[0][i]:.2f}")

print(f"\n🔬 Diagnóstico Real: {etiqueta_real.upper()}\n")

print(f"{'Modelo':<20} {'Predicción':<15} {'Confianza':<15} {'Estado'}")
print(f"{'-' * 70}")

for nombre, modelo in modelos.items():
    pred = modelo.predict(caso_escalado)[0]
    pred_nombre = datos.target_names[pred]
    proba = modelo.predict_proba(caso_escalado)[0]
    confianza = proba[pred]
    correcto = "✓" if pred == y_test[0] else "✗"
    print(f"{nombre:<20} {pred_nombre:<15} {confianza:<14.1%} {correcto}")

print(f"\n{'=' * 80}")
print("🏆 RESUMEN FINAL")
print(f"{'=' * 80}\n")

print(f"✨ Mejor Modelo: {mejor_modelo_nombre}")
print(f"   • Precisión en Test: {resultados[mejor_modelo_nombre]['accuracy']:.2%}")
print(f"   • Validación Cruzada: {resultados[mejor_modelo_nombre]['cv_mean']:.2%} (±{resultados[mejor_modelo_nombre]['cv_std']:.2%})")
print(f"   • AUC-ROC: {resultados[mejor_modelo_nombre]['roc_auc']:.3f}")

print(f"\n📋 Ranking de Modelos:")
ranking = sorted(resultados.items(), key=lambda x: x[1]['accuracy'], reverse=True)
for i, (nombre, metricas) in enumerate(ranking, 1):
    print(f"   {i}. {nombre}: {metricas['accuracy']:.2%}")

print(f"\n💡 Conclusión:")
print(f"   Este sistema de clasificación alcanza una precisión de {max(accuracies):.1%},")
print(f"   lo que lo hace muy confiable para asistir en el diagnóstico temprano")
print(f"   de cáncer de mama. Los modelos ensemble (Random Forest, Gradient Boosting)")
print(f"   tienden a ofrecer los mejores resultados en este tipo de problema médico.")

print(f"\n{'=' * 80}")
print("✅ Análisis completo finalizado. Revisa los gráficos generados.")
print(f"{'=' * 80}\n")
