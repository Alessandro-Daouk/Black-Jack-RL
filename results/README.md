# Index des expériences

## Résultats à lire pour le rendu intégré

- [`supervised/report.md`](supervised/report.md) : modèle supervisé livré, un train/validation/test.
- [`supervised/learning_curve/report.md`](supervised/learning_curve/report.md) : cinq jeux indépendants,
  quatre tailles train, comparaison avec les fréquences ; CSV, matrices, métadonnées et figure.
- [`hybrid/agent_tuned/report.md`](hybrid/agent_tuned/report.md) : agent livré, graine RL 0,
  évaluation exacte et 10 000 parties de graine 2027.
- [`hybrid/config_vs_tuned/report.md`](hybrid/config_vs_tuned/report.md) : sélection du réglage
  sur validation, puis comparaison sur 20 graines de test ; 160 entraînements.
- [`hybrid/frequencies_ablation/report.md`](hybrid/frequencies_ablation/report.md) : nouvelle
  comparaison régression / fréquences / exact / standard, 80 entraînements sur les graines 400–419.

Les résultats chiffrés sont dans les CSV/JSON ; les rapports expliquent leur portée.
Les anciennes mesures ne sont pas modifiées pour ajouter rétroactivement des expériences.
Les rapports textuels peuvent être clarifiés sans changer les mesures brutes.

## Références historiques conservées

- `metrics.csv`, `learning_curve.png`, `ablation.png`, `tuning.csv` : résultats
  du notebook et du réglage de Wassim, issus de sa branche `58f5ce3`.
- `hybrid/training.json`, `hybrid/evaluation.json`, `hybrid/report.md`, `hybrid/learning_curve.csv` :
  premier hybride supervisé CONFIG, associé à `models/hybrid/agent.npz`.
- `supervised/integration.json` et `integration_58f5ce3.json` : contrôles d'intégration historiques.

Le coût et les graines diffèrent selon les expériences : ne pas fusionner leurs lignes
comme s'il s'agissait d'une seule campagne. Les expériences de reproduction locales
peuvent être placées dans `results/reproduction/`.
