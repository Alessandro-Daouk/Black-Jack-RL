# Évaluation de l'agent hybride

> Rapport généré par la version historique du projet ; ses mesures sont inchangées. Certaines interprétations ont été corrigées depuis : en cas de désaccord, le [README](../../../README.md) et le [rapport scientifique](../../../docs/rapport_scientifique.md) font foi.

Le prédicteur supervisé reste figé pendant le Q-learning. Les sources RL de
Wassim proviennent du commit `58f5ce35880ae18c107187f4b64f23f2fa005ded`.

## Protocole

- RL : 100,000 épisodes, graine 0.
- Réglage RL : `tuned` ; eps0=1, eps_min=1,
  decay=0.99995, omega=1, gamma=1.
- Évaluation : 10,000 parties, graine 2027, sans exploration.
- Règles : Blackjack-v1, natural=False, sab=False ; récompenses -1/0/+1.
- Données du prédicteur : 60,000 mains
  d'entraînement ; validation et test indépendants, détaillés dans `training.json`.

## Résultats de simulation

- Victoires : 4,225 (42.25%).
- Égalités : 1,005 (10.05%).
- Défaites : 4,770 (47.70%).
- Gain moyen observé : -0.054500 par manche.
- Intervalle normal approximatif à 95 % : [-0.073059, -0.035941].

Cet intervalle ne mesure que le bruit de simulation pour l'agent fixé ; il ne
mesure pas les différences entre plusieurs entraînements.

## Évaluation exacte de la politique

- Gain moyen exact : -0.047206653 par manche.
- Gain optimal exact : -0.046555977.
- Écart à l'optimum : 0.000650676 unité de récompense par manche.

Le calcul exact utilise la récurrence de Wassim et la distribution initiale de
Gymnasium. Le gain observé peut dépasser le gain optimal exact par fluctuation
d'échantillonnage : cela ne signifie pas que la politique bat l'optimum.
Même la politique optimale a une espérance négative dans ces règles.

## Vérifications et limites

- STAND est resté figé pendant l'apprentissage.
- HIT a été observé dans 280 des 280 états atteignables.
- 0 décision HIT sans observation préalable de cette action pendant le test.
- Empreinte du prédicteur : `b91c797fcffa18ef421f73fff0ef7c088bcdacbad0d787d3ec819f67a9ebbfc1`.
- Empreinte de l'agent : `f8774e458407f4f97d7611c125ad1fda09bfae633d866e4e7db8c370a88f55df`.

`training.json` contient les paramètres, versions et empreintes des sources RL.
`learning_curve.csv` contient les évaluations exactes pendant l'entraînement.
Ce rapport décrit un seul entraînement RL : il ne suffit pas à comparer des méthodes.
La comparaison sur plusieurs graines des réglages CONFIG et TUNED, avec le Q-learning
standard et l'hybride exact comme références, est dans `results/hybrid/config_vs_tuned/`.
