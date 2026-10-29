# Reproduire les expériences

Toutes les commandes partent de la racine du dépôt. Installer avec `uv sync --locked`.
Les modèles et résultats livrés permettent de lire le rapport et jouer immédiatement.
Les sorties ci-dessous utilisent de nouveaux chemins pour préserver les artefacts livrés.
Les durées dépendent de l'ordinateur ; les grandes campagnes peuvent prendre plusieurs minutes.

## Contrôles rapides

```bash
uv run --locked pytest -q
uv run --locked python -m scripts.verify_release
uv run --locked python -m src.play --games 5 --seed 2030 --trace
```

## Supervisé : reproduire l'entraînement retenu

```bash
uv run --locked python -m src.supervised.train --model-dir models/reproduction/supervised --results-dir results/reproduction/supervised
```

Paramètres par défaut : train 60 000/graine 42, validation 12 000/43, test 12 000/123.
C choisi sur validation parmi 0.01, 0.1, 1, 10, 100 ; pas de refit. Cette commande
écrit le modèle, ses métadonnées et ses mesures. Ne pas réutiliser ces graines de test
pour choisir de nouvelles configurations : ce sont désormais des résultats connus.

## Hybride : reproduire le réglage retenu

```bash
uv run --locked python -m src.train_hybrid --preset tuned --agent models/reproduction/agent_tuned.npz --results-dir results/reproduction/hybrid --episodes 100000 --seed 0
uv run --locked python -m src.evaluate_hybrid --agent models/reproduction/agent_tuned.npz --output results/reproduction/hybrid/evaluation.json --games 10000 --seed 2027
```

Ces commandes utilisent le prédicteur livré par défaut. Pour utiliser le prédicteur
reproduit à l'étape précédente, ajouter `--model models/reproduction/supervised/dealer_model.joblib`
aux deux commandes. Chaque Q-table doit être chargée avec le fichier exact de son
prédicteur : le chargeur vérifie son SHA-256.

`--agent` est obligatoire. Si la Q-table ou son dossier de résultats existe déjà,
l'entraînement refuse de l'écraser. `--overwrite` autorise explicitement ce remplacement
et supprime alors les anciens `evaluation.json` et `report.md` de ce dossier : il faut réévaluer.
La sérialisation peut dépendre des versions ; comparer les tables et les scores,
et pas uniquement le binaire d'un modèle réentraîné sur une autre plateforme.

## Comparaison CONFIG/TUNED historique

```bash
uv run --locked python -m src.compare_hybrid --output-dir results/reproduction/config_vs_tuned --jobs 2
```

40 entraînements de validation (graines 200–219, deux réglages), puis 120 de test
(graines 0–19, trois méthodes et deux réglages), à 100 000 épisodes chacun.
Le réglage retenu utilise uniquement les scores de validation. Les résultats publiés
sont dans `results/hybrid/config_vs_tuned/`. Le script historique remplace les fichiers
de son dossier de sortie : conserver le chemin distinct proposé ci-dessus.

## Courbes supervisées et ablation des fréquences

```bash
uv run --locked python -m src.supervised.learning_curve --output results/reproduction/learning_curve --jobs 2
uv run --locked python -m src.compare_frequencies --output results/reproduction/frequencies_ablation --jobs 2
```

La première commande simule cinq jeux indépendants, avec quatre tailles train et
cinq C candidats par taille. La seconde lance 80 entraînements RL sur les graines
400–419. Les dossiers de sortie doivent être vides. `--jobs 1` permet une exécution
séquentielle. Les paramètres et limites figurent dans `docs/protocole_complementaire.md`.

Les figures supervisées sont générées automatiquement. Pour recréer la figure de
l'ablation **publiée**, depuis les CSV livrés :

```bash
uv run --locked python -m scripts.make_summary_figures
```

## Travail original de Wassim

- `src/tune.py` : recherche de eps_min et omega sur les graines 100–102, avec loi
  exacte pour l'hybride. `uv run --locked python -m src.tune` relance 108 entraînements,
  utilise les cœurs disponibles et remplace `results/tuning.csv`.
- `train_model2.ipynb` : standard, hybride exact et ablations avec probabilités
  empiriques simulées par le module de Wassim. Le notebook remplace `results/metrics.csv`,
  ses figures et `models/q_hit_table.pkl` ; le modèle intégré est dans d'autres dossiers.
- `uv sync --locked --extra notebook` installe le noyau pour l'éditeur de notebooks.
  Le projet n'impose pas de serveur Jupyter pour utiliser les modèles en ligne de commande.
- `docs/modele2.tex` et `docs/blackjack.pdf` : rapport original du modèle 2, conservé.

Pour reproduire intégralement ces sorties historiques, utiliser une copie dédiée du dépôt.
Le rapport intégré `docs/rapport.md` fait autorité pour la conclusion commune.
