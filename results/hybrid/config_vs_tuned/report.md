# Comparaison CONFIG / TUNED de l'hybride supervisé

> Rapport généré par la version historique du projet ; ses mesures sont inchangées. Certaines interprétations ont été corrigées depuis : en cas de désaccord, le [README](../../../README.md) et le [rapport scientifique](../../../docs/rapport_scientifique.md) font foi.

Sources RL de Wassim : commit `58f5ce35880ae18c107187f4b64f23f2fa005ded`.
Prédicteur supervisé figé : `b91c797fcffa18ef421f73fff0ef7c088bcdacbad0d787d3ec819f67a9ebbfc1`.

## Question et réglages

À budget RL égal, l'agent hybride qui utilise notre V_STAND estimée apprend-il mieux
avec la configuration précédente `CONFIG` ou avec `TUNED["hybrid"]` du commit 58f5ce3 ?

| Réglage | eps0 | eps_min | decay | omega | gamma |
|---|---:|---:|---:|---:|---:|
| CONFIG (précédent) | 1 | 0.05 | 0.99995 | 1 | 1 |
| TUNED["hybrid"] | 1 | 1 | 0.99995 | 1 | 1 |

Avec eps_min = 1, le comportement pendant l'apprentissage est entièrement aléatoire ; la politique
évaluée reste gloutonne, le Q-learning étant hors politique. Wassim a réglé TUNED avec la V_STAND
exacte (graines 100–102) : ce réglage n'était pas encore vérifié avec notre V_STAND estimée.

## Protocole

- Budget : 100,000 épisodes RL par entraînement, identique pour tous les bras.
- Les deux réglages de l'hybride supervisé partagent le même prédicteur, donc le même coût de
  simulation du croupier : 60,000 mains d'entraînement, plus 12,000 de
  validation et 12,000 de test pour le choisir et l'évaluer.
- Validation : graines 200–219, hybride supervisé seulement.
  Règle fixée avant l'exécution : meilleur gain exact moyen de l'hybride supervisé sur les graines de validation ; égalité → config ; le test ne sert à aucun choix..
- Test : graines 0–19, sans aucun choix. Références au même budget RL :
  Q-learning standard (CONFIG ou TUNED["standard"], eps_min = 0.4) et hybride avec V_STAND exacte.
- Métrique : gain exact J de la politique gloutonne finale (récurrence de Wassim, vraie loi du croupier).
  Elle est sans bruit pour une politique donnée ; la variabilité vient de l'entraînement RL.
- Statistiques : différences par graine ; test exact de permutation des signes, bilatéral..

## Validation et choix

| Méthode | Réglage | eps_min | J moyen | écart-type | écart à l'optimum | écart à la limite | états ≠ π* |
|---|---|---:|---:|---:|---:|---:|---:|
| Hybride, V_STAND supervisée | config | 0.05 | -0.048243 | 0.000790 | 0.001687 | 0.001665 | 5.5 |
| Hybride, V_STAND supervisée | tuned | 1 | -0.047222 | 0.000329 | 0.000666 | 0.000644 | 4.3 |

- TUNED − CONFIG : +0.001021 en moyenne (erreur type 0.000164) ; meilleur sur 20, égal sur 0, moins bon sur 0 des 20 graines ; p exact = 1.9e-06.

**Réglage retenu : `tuned`.**

## Test

| Méthode | Réglage | eps_min | J moyen | écart-type | écart à l'optimum | écart à la limite | états ≠ π* |
|---|---|---:|---:|---:|---:|---:|---:|
| Hybride, V_STAND supervisée | config | 0.05 | -0.048083 | 0.000995 | 0.001528 | 0.001506 | 5.2 |
| Hybride, V_STAND supervisée | tuned | 1 | -0.047347 | 0.000345 | 0.000791 | 0.000769 | 4.5 |
| Hybride, V_STAND exacte | config | 0.05 | -0.048624 | 0.001365 | 0.002068 | 0.002068 | 5.5 |
| Hybride, V_STAND exacte | tuned | 1 | -0.047371 | 0.000468 | 0.000815 | 0.000815 | 4.2 |
| Q-learning standard | config | 0.05 | -0.050667 | 0.001813 | 0.004111 | 0.004111 | 13.0 |
| Q-learning standard | tuned | 0.4 | -0.049459 | 0.000933 | 0.002904 | 0.002904 | 11.5 |

Différences appariées sur les graines de test (candidat − référence) :

- hybride supervisé, TUNED − CONFIG : +0.000737 en moyenne (erreur type 0.000209) ; meilleur sur 17, égal sur 0, moins bon sur 3 des 20 graines ; p exact = 0.0019.
- hybride exact, TUNED − CONFIG : +0.001253 en moyenne (erreur type 0.000304) ; meilleur sur 16, égal sur 0, moins bon sur 4 des 20 graines ; p exact = 0.00035.
- réglage tuned, hybride supervisé − hybride exact : +0.000024 en moyenne (erreur type 0.000073) ; meilleur sur 6, égal sur 5, moins bon sur 9 des 20 graines ; p exact = 0.77.
- hybride supervisé (tuned) − standard (TUNED) : +0.002113 en moyenne (erreur type 0.000236) ; meilleur sur 20, égal sur 0, moins bon sur 0 des 20 graines ; p exact = 1.9e-06.

Ces quatre comparaisons sont descriptives ; seule la validation a servi à décider.

### D'où vient l'écart de l'hybride supervisé

La politique limite, optimale pour notre V_STAND estimée, a un gain exact de -0.046577811.
Elle ne diffère de π* que sur 1 des 280 états atteignables ; l'écart dû au prédicteur vaut 0.000022.
La colonne « écart à la limite » est la différence de gain réel entre cette politique limite
et la politique apprise. Ce terme n'est pas garanti positif : l'optimum du modèle estimé
n'est pas une borne supérieure du gain dans le vrai environnement.

### Écart moyen à l'optimum selon le nombre d'épisodes RL (test)

| Méthode | Réglage | 10,000 | 20,000 | 50,000 | 100,000 |
|---|---|---:|---:|---:|---:|
| Hybride, V_STAND supervisée | config | 0.006924 | 0.003547 | 0.002256 | 0.001528 |
| Hybride, V_STAND supervisée | tuned | 0.006896 | 0.003738 | 0.001707 | 0.000791 |
| Hybride, V_STAND exacte | config | 0.007465 | 0.004228 | 0.002550 | 0.002068 |
| Hybride, V_STAND exacte | tuned | 0.006879 | 0.003524 | 0.001772 | 0.000815 |
| Q-learning standard | config | 0.019941 | 0.012562 | 0.006090 | 0.004111 |
| Q-learning standard | tuned | 0.019941 | 0.013200 | 0.005927 | 0.002904 |

### Budget total de simulations

Si l'on compte les mains simulées pour le prédicteur, un budget total de 100,000 simulations laisse à l'hybride supervisé 40,000 épisodes RL (mains d'entraînement seules) ou 16,000 (avec validation et test).

| Méthode | Réglage | Mains du croupier comptées | Épisodes RL | Écart moyen à l'optimum |
|---|---|---|---:|---:|
| Hybride, V_STAND supervisée | config | entraînement | 40,000 | 0.002264 |
| Hybride, V_STAND supervisée | tuned | entraînement | 40,000 | 0.001740 |
| Hybride, V_STAND supervisée | config | toutes | 16,000 | 0.004473 |
| Hybride, V_STAND supervisée | tuned | toutes | 16,000 | 0.004528 |
| Q-learning standard | config | aucune | 100,000 | 0.004111 |
| Q-learning standard | tuned | aucune | 100,000 | 0.002904 |
| Hybride, V_STAND exacte | config | aucune | 100,000 | 0.002068 |
| Hybride, V_STAND exacte | tuned | aucune | 100,000 | 0.000815 |

L'hybride exact utilise la loi analytique du croupier : son coût de simulation n'est pas compté,
c'est une référence avec probabilités exactes, pas une borne de performance à budget RL fini.
Seule la politique optimale π* fournit la borne supérieure du gain espéré.

## Vérifications

- STAND est resté figé dans tous les entraînements hybrides.
- Le run CONFIG de graine 0 reproduit bit à bit l'ancien agent `models/hybrid/agent_config.npz`.

## Limites

- 20 graines de test : les conclusions portent sur ce budget de 100,000 épisodes et ce prédicteur.
- Un seul prédicteur supervisé (une graine de données) : la variabilité due aux données
  supervisées n'est pas mesurée ici.
- Le test énumère exactement les permutations de signes des différences appariées. Sa validité
  suppose l'échangeabilité des deux traitements au sein des paires sous H0 (ou la symétrie des
  différences autour de zéro). L'égalité des moyennes seule ne suffit pas. Les p ne corrigent
  pas la multiplicité ; ces comparaisons sont descriptives et ne prouvent aucune équivalence.
- J utilise la vraie loi du croupier pour évaluer. L'hybride supervisé ne l'utilise pas pour
  ses mises à jour ; l'hybride exact, par définition, la reçoit pour calculer STAND.
- Le budget total additionne mains du croupier et épisodes RL : ce sont des unités de simulation
  différentes, pas une égalisation du temps de calcul. Le coût du réglage n'est pas inclus.

## Fichiers

- `runs.csv` : une ligne par entraînement (phase, méthode, réglage, graine, métriques).
- `learning_curves.csv` : gain exact tous les 2,000 épisodes pour chaque entraînement.
- `summary.json` : protocole, résumés, tests appariés, versions et empreintes.
