# Code

Tout se lance depuis la racine du dépôt avec `uv run --locked python -m <module>`. Les
modules sont regroupés ci-dessous par rôle ; les noms de fichiers n'ont pas été déplacés,
pour que la réplication de la première campagne porte sur le code d'origine.

## Le jeu et le modèle 2 (Wassim Ghnaya)

| Module | Rôle |
|---|---|
| `environment.py` | Règles de Blackjack-v1, paquet infini, transition de HIT |
| `stand_value.py` | Loi du croupier (exacte ou estimée) et valeur de STAND |
| `evaluate.py` | Référence exacte : Q*_HIT, politique optimale, gain exact J(π) |
| `q_learning.py` | Q-learning standard et hybride, réglages CONFIG et TUNED |
| `tune.py` | Réglage de ε_min et ω |

## Le modèle 1 : `supervised/`

| Module | Rôle |
|---|---|
| `data.py` | Simulation des mains du croupier |
| `model.py` | Régression logistique : ajustement, sauvegarde, chargement |
| `train.py` | Entraînement complet : sélection de C, export, évaluation |
| `evaluate.py`, `reference.py`, `plots.py` | Métriques, loi exacte de référence, figures |
| `values.py`, `constants.py` | Conversion des probabilités en table V_STAND |
| `learning_curve.py`, `demo.py` | Courbe d'apprentissage, démonstration |

## L'agent livré et la première campagne

| Module | Rôle |
|---|---|
| `hybrid_agent.py` | Agent hybride : charge le modèle 1 et Q_HIT, décide |
| `play.py` | Démonstration en ligne de commande |
| `export_web.py` | Exporte les modèles vers l'application web `web/` (jouer contre les modèles) |
| `train_hybrid.py`, `evaluate_hybrid.py` | Entraîner et évaluer un agent hybride |
| `compare_hybrid.py`, `compare_frequencies.py`, `experiment_utils.py` | Comparaisons de la première campagne |

## La campagne principale et ses compléments : `audit/`

| Module | Rôle | Résultats |
|---|---|---|
| `oracle.py` | Second oracle exact, écrit indépendamment de `evaluate.py` | |
| `core.py` | Simulation et entraînement instrumentés (temps, cartes), mêmes tirages que `q_learning.py` | |
| `pilot.py`, `campaign.py` | Pilote, puis campagne principale | `results/2_campagne_principale/` |
| `analyze.py` | Tableaux et figures du rapport | `results/2_campagne_principale/analyse/` |
| `diagnostics.py` | Contrôles de l'oracle, bornes, KL | `results/2_campagne_principale/diagnostics/` |
| `replicate.py` | Réplication de la première campagne | `results/2_campagne_principale/replication/` |
| `variant.py` | Cartes uniformes | `results/3_cartes_uniformes/` |
| `complements.py` | Planification, Double Q-learning | `results/4_autres_methodes/planification_double_q/` |
| `baselines.py` | Expected SARSA, Monte-Carlo, Dyna-Q, RL à modèle appris, variance | `results/4_autres_methodes/autres_algorithmes/` |

## L'extension en sabot fini : `shoe/`

| Module | Rôle |
|---|---|
| `game.py` | Sabot de 52 cartes, manche, loi exacte du croupier sachant les cartes non vues |
| `models.py` | Quatre modèles 1 (carte, table, logistique, boosting) et variables des agents linéaires |
| `campaign.py` | Modèles 1, réglage, agents tabulaires et linéaires → `results/5_sabot_fini/` |
| `figures.py` | Figure du §6 |
| `confirm.py` | Confirmation selon le plan figé → `results/6_confirmation_sabot/mesures/` |
