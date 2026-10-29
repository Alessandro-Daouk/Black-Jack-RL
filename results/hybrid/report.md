# Évaluation de l'agent hybride

Le prédicteur supervisé reste figé pendant le Q-learning. Les sources RL de
Wassim proviennent du commit `0780a4f32335540c5b59a602f38c08de3b3f01f6`.

## Protocole

- RL : 100,000 épisodes, graine 0.
- Évaluation : 10,000 parties, graine 2026, sans exploration.
- Règles : Blackjack-v1, natural=False, sab=False ; récompenses -1/0/+1.
- Données du prédicteur : 60,000 mains
  d'entraînement ; validation et test indépendants, détaillés dans `training.json`.

## Résultats de simulation

- Victoires : 4,368 (43.68%).
- Égalités : 942 (9.42%).
- Défaites : 4,690 (46.90%).
- Gain moyen observé : -0.032200 par manche.
- Intervalle normal approximatif à 95 % : [-0.050844, -0.013556].

Cet intervalle ne mesure que le bruit de simulation pour l'agent fixé ; il ne
mesure pas les différences entre plusieurs entraînements.

## Évaluation exacte de la politique

- Gain moyen exact : -0.047764775 par manche.
- Gain optimal exact : -0.046555977.
- Écart à l'optimum : 0.001208798 unité de récompense par manche.

Le calcul exact utilise la récurrence de Wassim et la distribution initiale de
Gymnasium. Le gain observé peut dépasser le gain optimal exact par fluctuation
d'échantillonnage : cela ne signifie pas que la politique bat l'optimum.
Même la politique optimale a une espérance négative dans ces règles.

## Vérifications et limites

- STAND est resté figé pendant l'apprentissage.
- HIT a été observé dans 280 des 280 états atteignables.
- 0 décision HIT sans observation préalable de cette action pendant le test.
- Empreinte du prédicteur : `b91c797fcffa18ef421f73fff0ef7c088bcdacbad0d787d3ec819f67a9ebbfc1`.
- Empreinte de l'agent : `52ecd4296c9bf039611558215b5e8c2cdae4d792323cd533a21502b21d9e1843`.

`training.json` contient les paramètres, versions et empreintes des sources RL.
`learning_curve.csv` contient les évaluations exactes pendant l'entraînement.
Cette expérience ne démontre pas un avantage sur le Q-learning standard.
Il reste à comparer plusieurs graines et des budgets explicites pour les agents
standard, hybride supervisé et hybride avec référence exacte.
