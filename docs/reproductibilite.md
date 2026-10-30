# Reproductibilité

Ce document décrit l'environnement dans lequel les résultats v2 ont été obtenus, les
commandes qui les régénèrent, leur durée mesurée et ce qui doit être identique ou
peut différer lors d'une reproduction.

## Environnement de référence

| Élément | Valeur |
|---|---|
| Système | Linux 7.0.0-38-generic, x86_64, glibc 2.43 |
| Processeur | Intel Core i7-1255U (2 cœurs performance, 8 cœurs efficacité, 12 fils), 30 Gio |
| Python | 3.12.7, installé par uv à partir de `.python-version` |
| Bibliothèques | NumPy 2.5.4, SciPy 1.18.1, scikit-learn 1.9.1, Gymnasium 1.4.0, pandas et joblib figés par `uv.lock` |
| LaTeX | pdfTeX 3.141592653-2.6-1.40.28 (TeX Live 2025, Debian), babel français, lmodern |

Chaque campagne écrit un `config.json` qui contient ses paramètres.

## Installation

```bash
uv sync --locked
```

`uv` doit être disponible dans le `PATH` ; sinon, utiliser son chemin d'installation.
La première installation télécharge Python et les dépendances ; ensuite, aucun accès
réseau n'est nécessaire.

## Modes de reproduction

**Mode léger** (quelques secondes, aucun entraînement) : `uv run --locked pytest -q`.

**Mode complet** : chaque commande refuse un dossier de sortie non vide. Les mesures de
temps n'ont de sens que si aucune autre charge lourde ne tourne pendant la campagne
finale ; les campagnes ont été exécutées l'une après l'autre.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
uv run --locked python -m src.audit.diagnostics --output results/reproduction/diagnostics
uv run --locked python -m src.audit.pilot --output results/reproduction/pilot
uv run --locked python -m src.audit.campaign --config experiments/v2/final_frozen.json --output results/reproduction/final
uv run --locked python -m src.audit.replicate --output results/reproduction/replication --jobs 6
uv run --locked python -m src.audit.variant --design experiments/v2/final_frozen.json --output results/reproduction/variant
uv run --locked python -m src.audit.analyze --input results/reproduction/final --diagnostics results/reproduction/diagnostics --variant results/reproduction/variant --replication results/reproduction/replication --output results/reproduction/analysis
```

Les compléments (E8, moins d'une minute) lisent la campagne finale
archivée `results/v2/final/` :

```bash
uv run --locked python -m src.audit.complements --output results/reproduction/complements
```

Sous PowerShell, remplacer la ligne `export` par
`$env:OMP_NUM_THREADS=1; $env:OPENBLAS_NUM_THREADS=1; $env:MKL_NUM_THREADS=1`.
La réplication régénère toutes les sorties des campagnes historiques ; le dépôt n'en
conserve que `comparison.json`, qui consigne les écarts aux résultats publiés, avec
`config.json` et `completion.json`.

La configuration `experiments/v2/final_frozen.json` a été produite par
`python -m src.audit.campaign --pilot results/v2/pilot --output <chemin>` à partir du
pilote archivé ; la régénérer n'est pas nécessaire, et la commande refuse d'écraser un
fichier existant.

**Campagnes historiques.** Elles se relancent individuellement :

```bash
uv run --locked python -m src.supervised.train --model-dir models/reproduction/supervised --results-dir results/reproduction/supervised
uv run --locked python -m src.train_hybrid --preset tuned --agent models/reproduction/agent_tuned.npz --results-dir results/reproduction/hybrid --episodes 100000 --seed 0
uv run --locked python -m src.evaluate_hybrid --agent models/reproduction/agent_tuned.npz --output results/reproduction/hybrid/evaluation.json --games 10000 --seed 2027
uv run --locked python -m src.compare_hybrid --output-dir results/reproduction/config_vs_tuned --jobs 2
uv run --locked python -m src.supervised.learning_curve --output results/reproduction/learning_curve --jobs 2
uv run --locked python -m src.compare_frequencies --output results/reproduction/frequencies_ablation --jobs 2
```

Les commandes `train_hybrid` et `evaluate_hybrid` utilisent le modèle 1 livré ; pour utiliser celui que la
première vient de reproduire, ajouter `--model models/reproduction/supervised/dealer_model.joblib`.
`uv run --locked python -m scripts.make_summary_figures` redessine, sans entraînement, la
figure publiée de l'ablation (`results/hybrid/frequencies_ablation/learning_curve.png`) à
partir de ses mesures. `uv run --locked python -m src.tune`
relance les 108 entraînements du réglage et réécrit `results/q_learning/tuning.csv`. Le
notebook `train_model2.ipynb` s'ouvre après `uv sync --locked --extra notebook` ; son
exécution réécrit les fichiers de `results/q_learning/` et `models/q_learning/`.

## Durées mesurées

| Étape | Durée murale | Remarque |
|---|---:|---|
| Diagnostics (oracle, baselines Monte-Carlo, KL, bornes) | 22 s | un processus |
| Pilote de validation (16 entraînements) | 80 s | un processus |
| Campagne finale (418 entraînements, 76 millions d'épisodes) | 62 min | un processus, un fil numérique ; 61 min de Q-learning |
| Réplication historique (240 entraînements et modèle 1) | 9,7 min | 6 processus parallèles |
| Variante à cartes uniformes (54 entraînements) | 3,6 min | un processus |
| Analyse et figures | 6 s | |
| Tests | quelques secondes | 113 tests |
| Compilation des deux PDF | quelques secondes | deux passes chacun |

## Ce qui doit être identique, et ce qui peut différer

- **Identique** à toute machine utilisant `uv.lock` : les mains simulées, les comptes de
  cartes, les trajectoires d'apprentissage, les Q-tables et donc les gains exacts.
  Les entraînements sont déterministes ; la réplication historique le vérifie à 10⁻⁹ près.
- **Différent** : toutes les durées, le pic de mémoire, ainsi que les
  analyses de coût à budget de temps fixé, puisque le point de grille retenu dépend du
  temps mesuré. Le processeur est hybride (cœurs performance et efficacité) : sans
  épinglage, l'ordonnanceur peut changer de type de cœur ; la dispersion observée du
  temps par épisode reste de 3 % (écart-type relatif) sur les 342 entraînements longs.
- **Possiblement différent** : les fichiers binaires sérialisés par joblib avec d'autres
  versions de bibliothèques ; comparer alors les probabilités et les gains.

## Régénérer les tableaux et documents

`src.audit.analyze` relit uniquement les fichiers bruts et régénère `summary.json`,
`surface.json`, `cost_contrasts.json`, les CSV dérivés, les figures et `report.md` de
`results/v2/analysis/`. Les chiffres du README et du rapport scientifique proviennent de
ces fichiers. `python scripts/build_docs.py` recompile les deux PDF (deux passes,
échec en cas de référence indéfinie ou de débordement) ; `--check` compile dans un
dossier temporaire sans remplacer les PDF publiés. Variables d'environnement fixées par
le script : `SOURCE_DATE_EPOCH` au 30 octobre 2026.
