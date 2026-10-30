# Ablation : remplacer la régression par les fréquences

Les deux estimateurs utilisent exactement les mêmes 60,000 mains d'entraînement du prédicteur figé.
Le budget RL et les réglages sont identiques entre hybrides ; le standard utilise TUNED[standard].
La référence exacte reçoit la vraie loi pour STAND. Elle n'est pas une borne à budget fini.

| Méthode | J moyen | Écart-type entre graines | Écart moyen à l'optimum |
|---|---:|---:|---:|
| logistic | -0.04734860 | 0.00057511 | 0.00079262 |
| frequencies | -0.04736438 | 0.00056830 | 0.00080840 |
| exact | -0.04727247 | 0.00043551 | 0.00071649 |
| standard | -0.04940536 | 0.00110089 | 0.00284939 |

## Différences appariées

- logistic_minus_frequencies : +0.00001579, IC approximatif à 95 % [-0.00001973, +0.00005130] ; meilleur/égal/moins bon : 1/18/1.
- logistic_minus_exact : -0.00007613, IC approximatif à 95 % [-0.00022060, +0.00006835] ; meilleur/égal/moins bon : 5/5/10.
- logistic_minus_standard : +0.00205677, IC approximatif à 95 % [+0.00143341, +0.00268013] ; meilleur/égal/moins bon : 18/0/2.

## Portée

Les intervalles portent sur la variabilité des entraînements RL, conditionnellement au seul jeu supervisé fixé.
Ils utilisent l'approximation de Student sur les différences appariées et ne corrigent pas la multiplicité.
Un intervalle contenant zéro n'établit pas l'équivalence. Le contraste principal est logistic − frequencies.
L'évaluation exacte J n'a pas de bruit de simulation pour une politique fixée.
Les 100,000 épisodes par entraînement n'incluent ni les mains supervisées ni le coût du réglage initial.
Graines : [400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419]. Elles ne servent pas à changer le modèle livré.

Reproduction : `python -m src.compare_frequencies`. Paramètres effectifs dans `protocol.json`.
Mesures par entraînement dans `runs.csv`, trajectoires dans `learning_curves.csv`, agrégats dans `summary.json`.
