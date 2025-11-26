import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import warnings
from utils import TextCleaner
warnings.filterwarnings('ignore')

# =============================================================================
# 1. CHARGEMENT ET EXPLORATION DU DATASET
# =============================================================================
print("="*80)
print("1. CHARGEMENT ET EXPLORATION DU DATASET")
print("="*80)

# Charger le dataset avec séparateur point-virgule
df = pd.read_csv('CHD.csv', sep=';')

# Afficher les premières lignes
print("\nPremières lignes du dataset:")
print(df.head())

# Afficher les types et informations générales
print("\nInformations générales:")
print(df.info())

# Distribution de famhist
print("\nDistribution de la variable famhist:")
print(df['famhist'].value_counts())

# Visualiser les valeurs manquantes
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
plt.title('Valeurs manquantes dans le dataset')
plt.tight_layout()
plt.savefig('missing_values.png')
print("\nHeatmap des valeurs manquantes sauvegardée: missing_values.png")

# =============================================================================
# 2. SÉPARATION DU DATASET
# =============================================================================
print("\n" + "="*80)
print("2. SÉPARATION DU DATASET")
print("="*80)

# Définir X et y
X = df.drop('chd', axis=1)
y = df['chd']

# Division train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=123, stratify=y
)

print(f"\nTaille de l'ensemble d'apprentissage: {X_train.shape}")
print(f"Taille de l'ensemble de test: {X_test.shape}")

# =============================================================================
# 3. PRÉTRAITEMENT DES VARIABLES NUMÉRIQUES
# =============================================================================
print("\n" + "="*80)
print("3. PRÉTRAITEMENT DES VARIABLES NUMÉRIQUES")
print("="*80)

# Pipeline pour variables numériques
numeric_features = ['sbp', 'ldl', 'adiposity', 'obesity', 'age']

numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # Imputation par la médiane
    ('scaler', StandardScaler())  # Standardisation
])

print("\nPipeline numérique créé:")
print("  1. SimpleImputer: Remplace les valeurs manquantes par la médiane")
print("  2. StandardScaler: Normalise les données (moyenne=0, écart-type=1)")

# =============================================================================
# 4. PRÉTRAITEMENT DE LA VARIABLE CATÉGORIELLE
# =============================================================================
print("\n" + "="*80)
print("4. PRÉTRAITEMENT DE LA VARIABLE CATÉGORIELLE")
print("="*80)

# Fonction pour uniformiser famhist (importée depuis utils.py)

# Pipeline pour variable catégorielle
categorical_features = ['famhist']

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Imputation
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))  # One-Hot Encoding
])

print("\nPipeline catégoriel créé:")
print("  1. SimpleImputer: Remplace les valeurs manquantes par la modalité la plus fréquente")
print("  2. OneHotEncoder: Encode les catégories en variables binaires")

# =============================================================================
# 5. CONSTRUCTION D'UN PRÉPROCESSEUR COMPLET
# =============================================================================
print("\n" + "="*80)
print("5. CONSTRUCTION DU PRÉPROCESSEUR COMPLET")
print("="*80)

# ColumnTransformer
preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_features),
    ('cat', categorical_pipeline, categorical_features)
])

print("\nColumnTransformer créé combinant les deux pipelines")

# =============================================================================
# 6. MODÈLE SUPERVISÉ AVEC ACP
# =============================================================================
print("\n" + "="*80)
print("6. MODÈLE AVEC ACP + RÉGRESSION LOGISTIQUE")
print("="*80)

# Pipeline complet avec ACP
pipeline_pca = Pipeline([
    ('cleaner', TextCleaner()),
    ('preprocessor', preprocessor),
    ('pca', PCA(n_components=0.95)),  # 95% de variance
    ('classifier', LogisticRegression(random_state=123, max_iter=1000))
])

# Entraîner le modèle
pipeline_pca.fit(X_train, y_train)

# Prédictions
y_pred_pca = pipeline_pca.predict(X_test)

# Évaluation
print("\nRapport de classification (avec ACP):")
print(classification_report(y_test, y_pred_pca))

accuracy_pca = accuracy_score(y_test, y_pred_pca)
print(f"\nAccuracy: {accuracy_pca:.4f}")

# =============================================================================
# 7. VARIANCE EXPLIQUÉE PAR L'ACP
# =============================================================================
print("\n" + "="*80)
print("7. ANALYSE DE LA VARIANCE EXPLIQUÉE PAR L'ACP")
print("="*80)

# Récupérer l'ACP du pipeline
pca = pipeline_pca.named_steps['pca']

# Variance expliquée
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print(f"\nNombre de composantes retenues: {pca.n_components_}")
print(f"Variance expliquée par composante: {explained_variance}")
print(f"Variance cumulative: {cumulative_variance}")

# Tracer la variance cumulative
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
plt.axhline(y=0.90, color='r', linestyle='--', label='90% de variance')
plt.xlabel('Nombre de composantes')
plt.ylabel('Variance expliquée cumulative')
plt.title('Variance expliquée par l\'ACP')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('pca_variance.png')
print("\nGraphique de variance sauvegardé: pca_variance.png")

# Nombre de composantes pour 90%
n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
print(f"\nNombre de composantes pour 90% de variance: {n_components_90}")

# Réentraîner avec n_components fixe si nécessaire
pipeline_pca_90 = Pipeline([
    ('cleaner', TextCleaner()),
    ('preprocessor', preprocessor),
    ('pca', PCA(n_components=n_components_90)),
    ('classifier', LogisticRegression(random_state=123, max_iter=1000))
])

pipeline_pca_90.fit(X_train, y_train)
y_pred_pca_90 = pipeline_pca_90.predict(X_test)
accuracy_pca_90 = accuracy_score(y_test, y_pred_pca_90)
print(f"\nAccuracy avec {n_components_90} composantes: {accuracy_pca_90:.4f}")

# =============================================================================
# 8. COMPARAISON AVEC UN MODÈLE SANS ACP
# =============================================================================
print("\n" + "="*80)
print("8. MODÈLE SANS ACP")
print("="*80)

# Pipeline sans ACP
pipeline_no_pca = Pipeline([
    ('cleaner', TextCleaner()),
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=123, max_iter=1000))
])

# Entraîner
pipeline_no_pca.fit(X_train, y_train)

# Prédictions
y_pred_no_pca = pipeline_no_pca.predict(X_test)

# Évaluation
print("\nRapport de classification (sans ACP):")
print(classification_report(y_test, y_pred_no_pca))

accuracy_no_pca = accuracy_score(y_test, y_pred_no_pca)
print(f"\nAccuracy: {accuracy_no_pca:.4f}")

# Comparaison
print("\n" + "-"*80)
print("COMPARAISON:")
print(f"  Accuracy avec ACP: {accuracy_pca:.4f}")
print(f"  Accuracy sans ACP: {accuracy_no_pca:.4f}")
if accuracy_pca > accuracy_no_pca:
    print("  → Le modèle avec ACP performe mieux")
else:
    print("  → Le modèle sans ACP performe mieux")

# =============================================================================
# 9. TEST D'UN MODÈLE KNN
# =============================================================================
print("\n" + "="*80)
print("9. MODÈLE KNN AVEC SMOTE")
print("="*80)

# Pipeline avec SMOTE et KNN
pipeline_knn = ImbPipeline([
    ('cleaner', TextCleaner()),
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=123)),
    ('pca', PCA(n_components=n_components_90)),
    ('classifier', KNeighborsClassifier())
])

# GridSearch pour optimiser n_neighbors
param_grid = {
    'classifier__n_neighbors': [3, 5, 7, 9, 11, 15, 20]
}

grid_search = GridSearchCV(
    pipeline_knn,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

print("\nRecherche des meilleurs hyperparamètres...")
grid_search.fit(X_train, y_train)

print(f"\nMeilleurs paramètres: {grid_search.best_params_}")
print(f"Meilleur score CV: {grid_search.best_score_:.4f}")

# Meilleur modèle KNN
best_knn = grid_search.best_estimator_

# Évaluation sur test
y_pred_knn = best_knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)

print("\nRapport de classification (KNN):")
print(classification_report(y_test, y_pred_knn))
print(f"\nAccuracy: {accuracy_knn:.4f}")

# Comparaison finale
print("\n" + "="*80)
print("COMPARAISON FINALE DES MODÈLES:")
print("="*80)
print(f"  Régression Logistique avec ACP: {accuracy_pca:.4f}")
print(f"  Régression Logistique sans ACP:  {accuracy_no_pca:.4f}")
print(f"  KNN avec SMOTE et ACP:          {accuracy_knn:.4f}")

# Sélectionner le meilleur modèle
models = {
    'LogReg_PCA': (pipeline_pca, accuracy_pca),
    'LogReg_NoPCA': (pipeline_no_pca, accuracy_no_pca),
    'KNN': (best_knn, accuracy_knn)
}

best_model_name = max(models, key=lambda k: models[k][1])
best_model = models[best_model_name][0]
best_accuracy = models[best_model_name][1]

print(f"\n🏆 MEILLEUR MODÈLE: {best_model_name} (Accuracy: {best_accuracy:.4f})")

# =============================================================================
# 10. ENTRAÎNEMENT FINAL ET SAUVEGARDE
# =============================================================================
print("\n" + "="*80)
print("10. ENTRAÎNEMENT FINAL ET SAUVEGARDE")
print("="*80)

# Entraîner sur toutes les données
print("\nEntraînement du meilleur modèle sur toutes les données...")
best_model.fit(X, y)

# Sauvegarder
joblib.dump(best_model, 'Model.pkl')
print("\n✅ Modèle sauvegardé: Model.pkl")

print("\n" + "="*80)
print("ANALYSE TERMINÉE!")
print("="*80)
print("\nFichiers générés:")
print("  - Model.pkl (modèle sauvegardé)")
print("  - missing_values.png (heatmap)")
print("  - pca_variance.png (variance expliquée)")
print("\nProchaine étape: Lancer l'application Streamlit avec 'streamlit run app.py'")