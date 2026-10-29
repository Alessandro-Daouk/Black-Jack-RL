# Blackjack : prédiction supervisée et Q-learning

Projet d'Alessandro et Wassim. **Question : estimer les probabilités du croupier
permet-il à un agent hybride d'apprendre une bonne politique avec moins d'épisodes RL ?**

L'agent livré associe une **régression logistique multinomiale**, qui prédit les
probabilités de `17, 18, 19, 20, 21, BUST`, à un **Q-learning tabulaire** qui apprend
la valeur de HIT. Il calcule la valeur de STAND à partir des probabilités prédites,
puis choisit l'action de plus grande valeur. Le modèle supervisé reste figé pendant le RL.

Règles : `Gymnasium Blackjack-v1`, `natural=False`, `sab=False`, paquet infini,
arrêt du croupier sur 17 souple, actions HIT/STAND, récompenses −1/0/+1.
Ni split, ni double, ni assurance, ni comptage des cartes.

## Essayer sur un autre ordinateur

Installer [Git](https://git-scm.com/downloads) et [uv](https://docs.astral.sh/uv/getting-started/installation/),
puis exécuter depuis un terminal. **La version intégrée est sur la branche ci-dessous** :

```bash
git clone --branch integration/supervised-hybrid https://github.com/Alessandro-Daouk/Black-Jack-RL.git
cd Black-Jack-RL
uv sync --locked
uv run --locked python -m scripts.verify_release
uv run --locked pytest -q
uv run --locked python -m src.play --games 5 --seed 2030 --trace
```

`uv` prépare Python 3.12.7 et l'environnement `.venv`. L'essai utilise les modèles
déjà entraînés ; aucun entraînement n'est requis. Le script de vérification contrôle
les empreintes des modèles, les sources RL et le gain exact annoncé.
Voir le [guide de validation sur un autre PC](docs/test_autre_pc.md).

Pour expliquer une situation `(total_joueur, carte_visible, as_utilisable)` :

```bash
uv run --locked python -m src.play --state 15 10 0
uv run --locked python -m src.supervised.demo --dealer 10 --player 15
```

La démonstration joue automatiquement et explique ses décisions. L'application web
interactive viendra après la validation du dépôt.

## Modèle retenu et résultats

Le couple livré est `models/supervised/dealer_model.joblib` +
`models/hybrid/agent_tuned.npz`. Le prédicteur utilise 60 000 mains train ; le RL,
100 000 épisodes, graine 0, paramètres `TUNED["hybrid"]` de Wassim.
La graine finale n'a pas été sélectionnée comme la meilleure graine de test.

- **Agent livré : gain exact −0,047207 par partie**, contre −0,046556 pour la politique optimale.
  Même l'optimum a une espérance négative avec ces règles.
- Campagne CONFIG/TUNED : sur 20 graines de test, l'hybride supervisé TUNED atteint
  −0,047347 en moyenne, contre −0,049459 pour le standard TUNED, à budget RL égal.
- Ce budget exclut la préparation du prédicteur. En comptant aussi ses 84 000 mains
  train/validation/test dans un budget total de 100 000 simulations, le standard repasse devant.
  Une main du croupier et un épisode RL ne représentent pas le même temps de calcul.
- Sur cinq jeux indépendants, l'excès exact moyen de log-loss du supervisé passe
  de 0,004507 à 5 000 mains à 0,000424 à 60 000, puis 0,000129 à 200 000.
  Les fréquences empiriques restent une référence compétitive ; le projet ne suppose
  pas une supériorité du classificateur.
- Ablation sur 20 nouvelles graines RL : différence régression−fréquences
  +0,00001579, IC approximatif [−0,00001973, +0,00005130] ; gains égaux sur 18 graines.
  Aucun avantage net de la régression n'est démontré.

Le [rapport complet](docs/rapport.md) rassemble la méthode, les résultats, les
incertitudes et les limites, y compris la nouvelle ablation sur 20 graines RL.
La [fiche du modèle](docs/model_card.md) précise le périmètre du modèle livré.

## Où lire et vérifier

- [Rapport scientifique](docs/rapport.md) : question, données, méthodes, métriques, résultats et limites.
- [Architecture de l'hybride](docs/agent_hybride.md) : calcul de STAND, apprentissage de HIT et décisions.
- [Protocole complémentaire](docs/protocole_complementaire.md) : expériences définies avant leur exécution.
- [Reproduire les expériences](docs/reproduction.md) : commandes et fichiers de sortie.
- [Index des résultats](results/README.md) : distinguer les expériences historiques et intégrées.
- [Provenance et contributions](docs/integration.md) : historique de Wassim, fichiers conservés et changements.

## Organisation

```text
src/supervised/             Simulation, classification, métriques, courbes d'apprentissage
src/q_learning.py           Q-learning standard et hybride de Wassim
src/stand_value.py          Probabilités exactes et valeurs de STAND de référence
src/evaluate.py             Évaluation exacte et politique optimale
src/hybrid_agent.py         Chargement du couple de modèles et décision
src/train_hybrid.py         Entraîner une nouvelle Q-table avec le prédicteur figé
src/compare_hybrid.py       Comparer CONFIG et TUNED
src/compare_frequencies.py  Ablation régression / fréquences / exact / standard
src/play.py                Démonstration locale
models/                    Modèles entraînés et manifeste des empreintes
results/                   Mesures brutes, rapports et figures
docs/                      Documentation scientifique et guides
tests/                     Tests des composants et de l'intégration
train_model2.ipynb          Notebook original de Wassim, conservé
```

Le notebook et `src/tune.py` sont utilisables avec les dépendances communes ; pour
ouvrir le notebook dans un éditeur compatible, installer aussi son noyau Python :
`uv sync --locked --extra notebook`. Leur périmètre et leurs sorties historiques
sont décrits dans le [guide de reproduction](docs/reproduction.md).

## Réutiliser le moteur

```python
from src.hybrid_agent import load_agent

agent = load_agent()                 # charger une seule fois
decision = agent.decide((15, 10, 0))
print(decision.action_name)          # HIT
print(decision.stand_value, decision.hit_value)
```

Cette interface pourra être appelée par l'application web. Modifier l'affichage
ne nécessite pas de réentraînement. Changer le prédicteur ou les règles du jeu
nécessite de réévaluer la cohérence du couple de modèles.
