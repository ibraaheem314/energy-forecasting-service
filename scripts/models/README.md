# Modèles Individuels

Architecture modulaire pour entraîner des modèles séparément avec cache optimisé.

## Modèles Disponibles

### Linear Regression
```bash
python scripts/models/linear_regression.py
# ou
make train-linear
```
- **Avantages** : Rapide, interprétable, baseline
- **Inconvénients** : Linéaire, pas de patterns complexes

### Random Forest  
```bash
python scripts/models/random_forest.py
# ou
make train-rf
```
- **Avantages** : Robuste, gère non-linéarité, feature importance
- **Inconvénients** : Plus lent, peut overfitter

### LightGBM
```bash
python scripts/models/lightgbm_model.py
# ou
make train-lgb
```
- **Avantages** : Très rapide, excellent performance, feature importance
- **Inconvénients** : Hyperparamètres sensibles

### **Gradient Boosting Quantile** (Recommandé)
```bash
python scripts/models/gradient_boosting_quantile.py
# ou
make train-gbq
```
- **Avantages** : **Prédictions par quantiles (10%, 25%, 50%, 75%, 90%)**
- **Métriques** : Coverage 80%, largeur intervalle de confiance
- **Use case** : Gestion des risques, intervalles de prédiction

## Entraîner Tous les Modèles
```bash
python scripts/run_all_models.py --data odre
# ou
make train-all
```

**OPTIMISATION** : Charge les données **UNE SEULE FOIS** et les partage entre tous les modèles !

## 🗂️ Système de Cache

### Infos Cache
```bash
python scripts/manage_cache.py info
make cache-info
```

### Précharger
```bash
python scripts/manage_cache.py preload --source odre
make cache-preload
```

### Vider Cache
```bash
python scripts/manage_cache.py clear
make cache-clear
```

## 📊 Résultats Typiques (ODRÉ 3 ans, 80k échantillons)

| Modèle | MAE | RMSE | R² | MAPE | Spécialité |
|--------|-----|------|----|----- |------------|
| **GBQuantile** | **2121** | **3723** | **0.716** | **4.70%** | 📊 **Quantiles** |
| Random Forest | 2136 | 3752 | 0.698 | 4.70% | 🌳 Robustesse |
| LightGBM | 2188 | 3825 | 0.686 | 4.86% | ⚡ Rapidité |
| Linear Reg | 2460 | 3820 | 0.687 | 5.41% | 📈 Baseline |

## 🎯 Use Cases

### 🔮 Prédictions Ponctuelles
→ `Random Forest` ou `LightGBM`

### 📊 Gestion des Risques
→ **`Gradient Boosting Quantile`** (intervalles de confiance)

### 📈 Baseline Rapide  
→ `Linear Regression`

### ⚡ Production Rapide
→ `LightGBM`

## 🏗️ Architecture

```
scripts/models/
├── base_model.py          # Classe de base commune
├── linear_regression.py   # 📈 Régression linéaire
├── random_forest.py       # 🌳 Forêt aléatoire
├── lightgbm_model.py      # ⚡ LightGBM
├── gradient_boosting_quantile.py  # 📊 GB Quantiles ⭐
└── __init__.py
```

Chaque modèle hérite de `BaseModel` et peut être lancé individuellement ou collectivement avec `run_all_models.py`.
