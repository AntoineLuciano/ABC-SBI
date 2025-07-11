# TODO List Simple - ABCNRE

## 🎯 Plan de développement en 3 étapes

### 1. 🔄 SIMULATION (code puis test)

#### Code à implémenter
- [ ] **ABCSimulator** (`src/abcnre/simulation/simulator.py`)
  - [ ] `generate_samples()` - ABC rejection sampling
  - [ ] `save_configuration()` - Sauvegarde config YAML
  - [ ] `load_configuration()` - Chargement config YAML
  - [ ] Gestion des données NPZ

- [ ] **RejectionSampler** (`src/abcnre/simulation/sampler.py`)
  - [ ] `sample()` - Algorithme ABC rejection
  - [ ] Calcul de distance/tolérance
  - [ ] Gestion des échantillons acceptés/rejetés

- [ ] **StatisticalModel base** (`src/abcnre/simulation/models/base.py`)
  - [ ] Interface abstraite pour modèles
  - [ ] `simulate()`, `prior_sample()`, `transform_phi()`

- [ ] **GaussGaussModel** (`src/abcnre/simulation/models/gauss_gauss.py`)
  - [ ] Implémentation modèle Gauss-Gauss
  - [ ] Migration du code existant

- [ ] **GAndKModel** (`src/abcnre/simulation/models/g_and_k.py`)
  - [ ] Implémentation modèle G-and-K
  - [ ] Migration du code existant

- [ ] **ConfigLoader** (`src/abcnre/config/loader.py`)
  - [ ] Chargement/validation YAML
  - [ ] Schémas de configuration

#### Tests à écrire
- [ ] **test_simulator.py**
  - [ ] Test initialisation ABCSimulator
  - [ ] Test generate_samples avec différents modèles
  - [ ] Test save/load configuration

- [ ] **test_sampler.py**
  - [ ] Test RejectionSampler.sample()
  - [ ] Test convergence et acceptance rate

- [ ] **test_models.py**
  - [ ] Test GaussGaussModel.simulate()
  - [ ] Test GAndKModel.simulate()
  - [ ] Test transform_phi()

---

### 2. 🧠 INFERENCE (code puis test)

#### Code à implémenter
- [ ] **NeuralRatioEstimator** (`src/abcnre/inference/estimator.py`)
  - [ ] `fit()` - Entraînement du NN
  - [ ] `predict_ratio()` - Prédiction du ratio
  - [ ] `approximate_posterior()` - Approximation postérieure
  - [ ] `save_model()` / `load_model()` - Persistance

- [ ] **DeepSetNetwork** (`src/abcnre/inference/networks/deepset.py`)
  - [ ] Architecture DeepSet pour NRE
  - [ ] Couches phi et rho
  - [ ] Forward pass

- [ ] **MLPNetwork** (`src/abcnre/inference/networks/mlp.py`)
  - [ ] Architecture MLP simple
  - [ ] Alternative à DeepSet

- [ ] **InferenceTrainer** (`src/abcnre/inference/trainer.py`)
  - [ ] Logique d'entraînement
  - [ ] Loss functions pour NRE
  - [ ] Optimisation et validation

- [ ] **Loss functions** (`src/abcnre/inference/losses.py`)
  - [ ] Binary cross-entropy pour ratio estimation
  - [ ] Autres losses possibles

#### Tests à écrire
- [ ] **test_estimator.py**
  - [ ] Test NeuralRatioEstimator.fit()
  - [ ] Test predict_ratio() avec données simulées
  - [ ] Test save/load model

- [ ] **test_networks.py**
  - [ ] Test DeepSetNetwork forward pass
  - [ ] Test MLPNetwork forward pass
  - [ ] Test dimensions input/output

- [ ] **test_trainer.py**
  - [ ] Test InferenceTrainer avec données ABC
  - [ ] Test convergence training
  - [ ] Test intégration avec simulator

---

### 3. 📊 DIAGNOSTIC (code puis test)

#### Code à implémenter
- [ ] **PosteriorValidator** (`src/abcnre/diagnostics/validator.py`)
  - [ ] `simulation_based_calibration()` - SBC complet
  - [ ] `coverage_analysis()` - Analyse de couverture
  - [ ] `compare_posteriors()` - Comparaison vraie vs approx

- [ ] **SBCAnalyzer** (`src/abcnre/diagnostics/sbc.py`)
  - [ ] Implémentation SBC détaillée
  - [ ] Statistiques de rang
  - [ ] Tests de calibration

- [ ] **Metrics** (`src/abcnre/diagnostics/metrics.py`)
  - [ ] RMSE, coverage probability
  - [ ] KL divergence, Wasserstein distance
  - [ ] Métriques custom pour ABC

- [ ] **Plots** (`src/abcnre/diagnostics/plots.py`)
  - [ ] Visualisation SBC
  - [ ] Comparaison postérieures
  - [ ] Rank plots, PP plots

#### Tests à écrire
- [ ] **test_validator.py**
  - [ ] Test PosteriorValidator.simulation_based_calibration()
  - [ ] Test coverage_analysis avec postérieures connues
  - [ ] Test compare_posteriors()

- [ ] **test_sbc.py**
  - [ ] Test SBCAnalyzer avec cas simples
  - [ ] Test détection de miscalibration
  - [ ] Test statistiques de rang

- [ ] **test_metrics.py**
  - [ ] Test calcul métriques diverses
  - [ ] Test avec distributions connues

---

## 🚀 Ordre d'exécution recommandé

### Phase 1: SIMULATION
1. Implémenter `StatisticalModel` base + `GaussGaussModel`
2. Implémenter `RejectionSampler` 
3. Implémenter `ABCSimulator`
4. Écrire tous les tests simulation
5. ✅ **Milestone 1**: Génération de données ABC fonctionnelle

### Phase 2: INFERENCE  
1. Implémenter `DeepSetNetwork`
2. Implémenter `NeuralRatioEstimator`
3. Implémenter `InferenceTrainer`
4. Écrire tous les tests inference
5. ✅ **Milestone 2**: Entraînement NRE fonctionnel

### Phase 3: DIAGNOSTIC
1. Implémenter `SBCAnalyzer`
2. Implémenter `PosteriorValidator`
3. Implémenter `Metrics` et `Plots`
4. Écrire tous les tests diagnostic
5. ✅ **Milestone 3**: Pipeline complet ABC-NRE

## 🎯 Objectif final
- [ ] **Pipeline end-to-end fonctionnel**: Simulation → Inference → Diagnostic
- [ ] **Tests complets**: Coverage > 80%
- [ ] **Exemple concret**: Cas Gauss-Gauss du début à la fin
- [ ] **Documentation**: README à jour avec exemples réels