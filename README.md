# Blackjack : une estimation du croupier aide-t-elle le Q-learning ?

Projet de Machine Learning 2 réalisé par Alessandro Daouk et Wassim Ghnaya.

**En une phrase.** Donner à un Q-learning la valeur de l'action « rester », estimée à
part à partir de mains de croupier simulées, lui fait atteindre en **10 000 parties** la
qualité que le Q-learning standard atteint en 40 000, et le place devant six autres
algorithmes qui apprennent en jouant. Cet avantage survit au coût de l'estimation, mais
il s'inverse si l'estimation est trop imprécise et que l'apprentissage dure assez
longtemps. En sabot fini, où la loi du croupier dépend des cartes déjà sorties, un modèle
supervisé sur la composition du sabot donne à l'agent une information que son état ne
contient pas, et l'hybride garde alors un avantage durable que les modèles plus pauvres
perdent.

## Le problème

Un agent d'apprentissage par renforcement apprend en jouant, et chaque partie coûte.
Au Blackjack, la valeur de l'action « rester » (STAND) ne dépend que de la main finale
du croupier : on peut l'estimer en simulant des mains de croupier, sans jouer.
**Question :** quand une composante de la fonction de valeur peut être estimée à part,
vaut-il mieux l'estimer et la fixer, ou laisser le renforcement tout apprendre ?

Chaque politique est jugée par son *regret* : l'écart entre son gain espéré par partie,
calculé exactement, et celui de la politique optimale (−0,0466 : le jeu est perdant
pour le joueur, et aucun agent de ce dépôt ne gagne d'argent).

## Installation et démonstration

Python 3.12.7 et [uv](https://docs.astral.sh/uv/getting-started/installation/), sous
Linux, macOS ou Windows. Aucun entraînement n'est nécessaire : les modèles sont fournis.

```bash
git clone https://github.com/Alessandro-Daouk/Black-Jack-RL.git
cd Black-Jack-RL
uv sync --locked
uv run --locked pytest -q
```

```bash
uv run --locked python -m src.play --games 5 --seed 2030 --trace   # cinq parties commentées
uv run --locked python -m src.play --state 15 10 0                 # une décision
```

```text
État (15, 10, 0) : STAND=-0.582872, HIT=-0.541728 -> HIT (HIT observé 1484 fois à l'entraînement)
```

Avec 15 points face à un 10, rester perd en moyenne 0,583 et tirer 0,542 : l'agent tire.

```python
from src.hybrid_agent import load_agent
agent = load_agent()
print(agent.decide((15, 10, 0)).action_name)   # HIT
```

## Méthode

Environnement `Blackjack-v1` de Gymnasium (paquet infini, croupier qui reste sur 17,
pas de doublement ni de partage). L'agent observe `(total, carte visible du croupier,
as utilisable)` : 280 états, au plus 20 décisions par partie, aucun état revisité.
Le jeu se résout donc exactement, ce qui sert d'**oracle** pour mesurer le gain de
chaque politique sans bruit de simulation.

- **Modèle 1 (supervisé)** : loi du résultat final du croupier sachant sa carte
  visible, par régression logistique multinomiale ou par simple table de fréquences.
  On en déduit $\hat V_{\mathrm{STAND}}$.
- **Modèle 2 (renforcement)** : le Q-learning standard apprend STAND et HIT ; l'hybride
  fixe STAND à $\hat V_{\mathrm{STAND}}$ et n'apprend que $Q_{\mathrm{HIT}}$.

Le prix de l'hybride est connu d'avance : il converge vers la meilleure politique **de
son modèle du croupier**, pas vers l'optimum vrai (démontré dans
[docs/modele2.pdf](docs/modele2.pdf)). L'architecture et ses justifications sont
détaillées dans le [rapport](docs/rapport_scientifique.md#2-cadre).

**Plan expérimental.** 28 jeux de données du croupier indépendants, deux graines
d'apprentissage par jeu ; intervalles à 95 % calculés entre jeux. Une différence
de 0,001 par partie est fixée comme seuil d'intérêt : c'est à peu près le coût d'une
seule décision fausse de la stratégie optimale, et 2,1 % de l'avantage de la maison.

## Résultats

![Regret en fonction des épisodes, du temps et des cartes](results/2_campagne_principale/analyse/learning_and_cost.png)

1. **L'hybride apprend beaucoup plus vite.** Il atteint en 10 000 parties le regret que
   le standard atteint en 40 000, et en 40 000 un regret que le standard n'atteint
   qu'après environ 200 000. À 20 000 mains et 100 000 parties, il gagne encore
   +0,00214 par partie [+0,00188 ; +0,00240], meilleur dans les 56 paires.
2. **Mais il a un plancher.** Son regret ne descend pas sous l'erreur de son modèle :
   environ 0,003 avec 1 000 mains, 0,0007 avec 5 000, 0,00013 avec 20 000. Avec
   1 000 mains, le standard le dépasse à 200 000 parties (−0,0020 [−0,0033 ; −0,0006]).
3. **Le coût de l'estimation ne l'efface pas.** Une main de croupier coûte environ un
   tiers de partie en temps. En secondes, acquisition comprise, l'hybride à 20 000
   mains reste devant pour des budgets de 2 à 20 s ; il perd si l'acquisition consomme
   le budget ou si le modèle est trop imprécis.
4. **La régression logistique n'apporte rien.** Une table de fréquences donne le même
   agent (équivalence établie à ±0,001) pour trois fois moins de temps : avec une seule
   variable catégorielle, la régression ne fait que reproduire les fréquences.
5. **Aucun autre algorithme qui apprend en jouant ne fait mieux.** Double Q-learning,
   Expected SARSA, contrôle Monte-Carlo et Dyna-Q, réglés comme le standard, font moins
   bien que lui ; un RL qui estime tout le modèle à partir des parties jouées et le résout
   exactement fait jeu égal. L'hybride les dépasse tous (de +0,0022 à +0,0103 à 100 000
   parties). Son avantage vient donc de l'information sur STAND, pas d'une meilleure
   exploitation des parties : STAND n'est observée que quand l'agent reste, avec un bruit
   de ±1.
6. **Une planification fait mieux, si l'on accepte de modéliser HIT.** Les mêmes mains
   de croupier permettent d'estimer la loi des cartes, et donc de calculer HIT sans
   jouer. Cette référence fait mieux que l'hybride après 100 000 parties
   (+0,00069 [+0,00057 ; +0,00081]). L'hybride n'est intéressant que lorsque seule
   l'action terminale peut être modélisée, ou quand son modèle exploite une information
   que l'état ne contient pas : c'est le cas du sabot fini (point 7).
7. **En sabot fini, le modèle supervisé compte enfin.** Avec un seul jeu de 52 cartes,
   la loi du croupier dépend des cartes déjà sorties. Une régression logistique sur la
   composition du sabot estime cette loi bien mieux qu'une table par classe de compte ou
   que le gradient boosting (KL vers la loi vraie : 0,0046 contre 0,019 et 0,010 à
   100 000 mains). Après 3 millions de parties, l'hybride qui l'utilise bat le Q-learning
   standard et les hybrides à table de +0,0009 par manche [+0,0004 ; +0,0014], dans les
   18 paires, alors que les hybrides à table ont rejoint le standard. En temps, ce gain
   n'est pas établi : l'ajustement de la logistique coûte plus que tout l'apprentissage
   du standard.

![Sabot fini : qualité du modèle 1 et gain des agents](results/5_sabot_fini/shoe.png)

![Différence de gain selon la taille des données et le budget RL](results/2_campagne_principale/analyse/budget_surface.png)

*Différence hybride − standard selon le nombre de mains du croupier et le budget de
parties (9 jeux) ; l'étoile signale un intervalle à 95 % qui exclut zéro.*

Les points 5 à 7 viennent d'analyses exploratoires, faites après la campagne principale. Le détail de tous
les résultats, les coûts mesurés et les ablations sont dans le
[rapport scientifique](docs/rapport_scientifique.md).

## Limites

- **Un seul jeu.** Trois variantes du Blackjack, sans double ni partage. Avec des cartes
  équiprobables, l'avantage n'est plus établi à 100 000 parties : le chiffre de 0,00214
  n'est pas une constante de la méthode. En sabot fini, il n'y a plus d'oracle exact :
  les gains sont estimés par Monte-Carlo apparié, et la mise ne varie pas.
- **Mécanisme non démontré.** On sait que fixer STAND cause l'écart (à exploration
  identique, les deux agents jouent les mêmes parties), pas pourquoi.
- **Temps mesurés** sur un ordinateur portable par campagne et une implémentation ; ceux
  du sabot fini, mesurés sous charge parallèle, sont indicatifs.
- **Analyses secondaires** sur 9 jeux, sans correction de multiplicité.

## Reproduire

Les tests ci-dessus prennent quelques secondes. La campagne complète
(environ 1 h 30) et les campagnes historiques sont décrites
dans [docs/reproductibilite.md](docs/reproductibilite.md). Les gains exacts sont
déterministes ; seules les durées varient. La CI exécute les tests et la
démonstration sous Linux et Windows.

## Organisation

Quatre dossiers à la racine, chacun avec son rôle :

| Dossier | Contenu | Guide |
|---|---|---|
| `docs/` | Rapport (**fait foi**), démonstrations PDF, état de l'art, reproductibilité | tableau ci-dessous |
| `src/` | Tout le code : jeu, modèle 1, modèle 2, agent, campagnes | [src/README.md](src/README.md) |
| `models/` | Modèles livrés, chargés par la démonstration | ci-dessous |
| `results/` | Mesures, numérotées dans l'ordre du rapport | [results/README.md](results/README.md) |

```text
results/
├── 1_premiere_campagne/      premiers modèles et comparaisons (q_learning, modele1, hybride)
├── 2_campagne_principale/    plan figé, entraînements, analyse, diagnostics, réplication
├── 3_cartes_uniformes/       robustesse à une autre loi des cartes
├── 4_autres_methodes/        planification, Double Q, Expected SARSA, Monte-Carlo, Dyna-Q
└── 5_sabot_fini/             extension où le modèle 1 utilise la composition du sabot

models/
├── supervised/               modèle 1 livré (régression logistique) et ses probabilités
├── hybrid/                   agent hybride livré (agent_tuned.npz) et l'ancien (agent_config.npz)
└── q_learning/               table Q_HIT produite par le notebook train_model2.ipynb
```

Le code est volontairement dupliqué à deux endroits : `src/audit/` réimplémente la
simulation et l'entraînement pour mesurer les coûts **sans changer les tirages
aléatoires** du code de la première campagne, et son oracle est écrit indépendamment
pour contrôler le premier. Les sources de Wassim Ghnaya sont intégrées sans
modification, pour que la réplication porte sur le code d'origine.

| Document | Contenu |
|---|---|
| [docs/rapport_scientifique.md](docs/rapport_scientifique.md) | Rapport complet : cadre, résultats, coûts, discussion (**fait foi**) |
| [docs/modele1.pdf](docs/modele1.pdf), [docs/modele2.pdf](docs/modele2.pdf) | Démonstrations : modèle 1, convergence et bornes de regret |
| [docs/etat_de_l_art.md](docs/etat_de_l_art.md) | Positionnement bibliographique |
| [docs/reproductibilite.md](docs/reproductibilite.md) | Environnement, commandes et durées |

## Auteurs et licence

- **Wassim Ghnaya** : environnement et références exactes, Q-learning standard et
  hybride, réglage, premières expériences (notebook), fondements du modèle 2.
- **Alessandro Daouk** : modèle 1 et son évaluation, assemblage de l'agent, campagnes
  de comparaison, reproductibilité, fondements du modèle 1, documentation commune.

Code sous licence [MIT](LICENSE). L'historique Git conserve l'auteur de chaque
contribution ; les versions antérieures des documents mathématiques sont au commit
`9c3d631`.
