# Résultats

Chaque dossier répond à une question du [rapport](../docs/rapport_scientifique.md) ; les
numéros suivent l'ordre du rapport. Les gains sont déterministes : relancer une commande
redonne les mêmes chiffres, seules les durées changent. Les commandes complètes sont dans
[docs/reproductibilite.md](../docs/reproductibilite.md).

| Dossier | Question | Rapport | Produit par |
|---|---|---|---|
| [1_premiere_campagne/](1_premiere_campagne/) | Premiers modèles : le modèle 1 est-il calibré, l'hybride apprend-il plus vite ? | §5.1 (répliquée) | `src.tune`, notebook, `src.supervised.*`, `src.compare_*` |
| [2_campagne_principale/](2_campagne_principale/) | Critère principal, budgets, coûts, ablations, sur 28 jeux indépendants | §4, §5.1–5.7 | `src.audit.*` |
| [3_cartes_uniformes/](3_cartes_uniformes/) | L'avantage tient-il avec une autre loi des cartes ? | §5.8 | `src.audit.variant` |
| [4_autres_methodes/](4_autres_methodes/) | L'hybride bat-il les autres algorithmes et la planification ? | §5.2, §5.3, §5.9 | `src.audit.complements`, `src.audit.baselines` |
| [5_sabot_fini/](5_sabot_fini/) | Le modèle 1 compte-t-il quand il voit ce que l'état RL ne voit pas ? | §6 | `src.shoe.campaign` |

## 1_premiere_campagne — les premiers modèles

Résultats obtenus avant la campagne principale ; leurs chiffres sont reproduits à
l'identique (§5.1), certaines interprétations ont été corrigées depuis : le rapport fait foi.

| Sous-dossier | Contenu |
|---|---|
| `q_learning/` | Notebook `train_model2.ipynb` : standard contre hybride à STAND exacte, réglage (`tuning.csv`) |
| `modele1/` | Modèle 1 livré : choix de C, calibration, erreur sur STAND, courbe d'apprentissage |
| `hybride/agent_tuned/`, `hybride/agent_config/` | Entraînement et évaluation des deux agents livrés |
| `hybride/config_vs_tuned/` | Comparaison des deux réglages RL |
| `hybride/frequencies_ablation/` | Logistique contre fréquences, première version |

## 2_campagne_principale — la mesure de référence

| Élément | Contenu |
|---|---|
| `plan_fige.json` | Plan figé avant la campagne : tailles, budgets, graines, critère principal |
| `pilote/` | Pilote qui a fixé le nombre de jeux (28) |
| `entrainements/` | Les 418 entraînements : `runs.csv`, `curves.csv`, modèles 1 (`fits/`), agents (`agents/`) |
| `analyse/` | Tous les tableaux et figures du rapport : **commencer par `report.md`** |
| `diagnostics/` | Oracle contre Monte-Carlo, bornes de regret, KL, sensibilité des états |
| `replication/` | Vérification que la première campagne se reproduit à l'identique |

## 3, 4, 5 — analyses complémentaires

Chaque dossier contient `config.json` (paramètres), `summary.json` (chiffres du
rapport), les CSV bruts et `completion.json` (statut et durée).

| Dossier | Fichiers principaux |
|---|---|
| `3_cartes_uniformes/` | `runs.csv`, `curves.csv`, `oracle_mc.json` (contrôle du nouvel oracle) |
| `4_autres_methodes/planification_double_q/` | `planning.csv`, `double_q.csv`, `tuning.csv` |
| `4_autres_methodes/autres_algorithmes/` | `curves.csv` (Expected SARSA, Monte-Carlo, Dyna-Q, RL à modèle appris), `variance_runs.csv` |
| `5_sabot_fini/` | `supervised.csv` (4 modèles 1), `runs.csv` (agents), `shoe.png` (figure du §6) |

Les `config.json` écrits avant cette organisation des dossiers peuvent contenir des noms
de modules ou d'options de l'époque ; les chemins de dossiers ont été mis à jour.
