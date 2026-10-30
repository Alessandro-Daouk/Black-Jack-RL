# Réponse au cahier des charges de l'audit

Chaque ligne reprend une demande du mandat d'audit scientifique, la vérification
effectuée, la modification qui en résulte, où la trouver, et son statut. « Fait » signifie
exécuté et vérifié ; « partiel » signale une limite explicitement décrite ; « non fait »
est justifié.

## Démarrage et préservation (§2)

| Demande | Vérification | Modification | Fichier ou résultat | Statut |
|---|---|---|---|---|
| Consigner commit, état, versions, matériel | Relevé avant toute modification | Inventaire et empreintes | `results/v2/baseline/environment.json`, `docs/audit_initial.md` | Fait |
| Exécuter les vérifications de base | 98 tests et `verify_release` passaient au départ | Aucune | `docs/audit_initial.md` | Fait |
| Audit initial hiérarchisé | Lecture intégrale du code et des documents | Tableau hypothèse → code → mesure → diagnostic | `docs/audit_initial.md` | Fait |
| Préserver modèles et résultats historiques | Empreintes contrôlées par `verify_release` et les tests | Résultats v2 dans `results/v2/`, scripts refusant d'écraser | `results/README.md` | Fait |
| Préserver le travail de Wassim Ghnaya | Empreintes de `docs/wassim_source.json` | Sources inchangées ; instrumentation dans `src/audit/` | `tests/`, `scripts/verify_release.py` | Fait |

## Mathématiques, modèle 1 (§3.1)

| Demande | Vérification | Modification | Fichier ou résultat | Statut |
|---|---|---|---|---|
| A. KL infini des fréquences brutes | Confirmé : P(case vide) > 0 à tout n fini | Proposition d'espérance infinie ; développement aléatoire avec loi limite χ² ; 25/n démontré pour Dirichlet lissé | `docs/modele1.tex` §5–6 | Fait |
| A. Validation numérique de 25/n | 2 000 tirages × 8 tailles, KL infinis comptés | Jeffreys et Laplace convergent vers 25 ; brut infini jusqu'à n = 1 000 | `results/v2/diagnostics/kl_summary.json`, `results/v2/analysis/kl_curve.png` | Fait |
| B. Objectif exact de scikit-learn | Source installée 1.9.1 inspecté | Pénalité ‖W‖²/(2Cn), biais non pénalisé ; convexité stricte sur le quotient ; relations stationnaires vérifiées (résidu 1,5·10⁻⁷) | `docs/modele1.tex` §3, `results/v2/diagnostics/sklearn_objective.json` | Fait |
| C. Scores, calibration, ECE | ECE non utilisée (discrétisation arbitraire) ; KL à la loi exacte et intervalles de Wilson non simultanés | Justification ajoutée ; KL non écrêté distingué de la log-loss écrêtée | `docs/modele1.tex` §2, `src/supervised/evaluate.py` | Fait |
| C. Pas d'équivalence par IC contenant zéro | Ancienne formulation « aucune différence » relevée | Marge 0,001 fixée a priori ; TOST ajouté (déviation D1) | `docs/protocole_v2_deviations.md`, `src/audit/analyze.py` | Fait |
| C. Test fixe contre nouveaux jeux | Les IC historiques conditionnent sur un prédicteur | 28 jeux supervisés indépendants en v2 | `docs/rapport_scientifique.md` §4 | Fait |

## Mathématiques, modèle 2 (§3.2)

| Demande | Vérification | Modification | Fichier ou résultat | Statut |
|---|---|---|---|---|
| MDP, états atteignables, terminaison | Exploration explicite des mains | 280 états ; horizon 20 décisions (et non 18 : deux as, total dur 2) | `docs/modele2.tex` §1, `tests/test_audit.py` | Fait |
| V_STAND exacte, estimée et Q*_HIT correspondants | Notations confondues dans l'ancienne version | Trois objets distincts : optimum vrai, optimum du modèle, table finie | `docs/modele2.tex` §2 | Fait |
| Convergence, hypothèses complètes | Σα = ∞, Σα² < ∞ insuffisants seuls | Lemme d'approximation stochastique, induction sur le DAG, visitation par ε ≥ ε_min > 0 ; références Tsitsiklis 1994, Jaakkola et al. 1994 | `docs/modele2.tex` §3 | Fait |
| Bornes 2TV, η, marge 2η, regret | Facteur 36η injustifié | Démontré : 2η ; borne locale pondérée par occupation ; borne d'agent fini 2η + δ·E[T] | `docs/modele2.tex` §5 | Fait |
| Limite asymptotique contre budget fini | Les bornes portaient sur la politique limite | Section « agent fini et limites des garanties » ; mesures : borne finie vraie mais lâche | `docs/modele2.tex` §5.2, `results/v2/analysis/finite_value_errors.csv` | Fait |
| Variance et biais de maximisation | Présentés comme garantis | Requalifiés en mécanismes plausibles ; analogie Expected Sarsa qualifiée | `docs/modele2.tex` §4 | Fait |
| Borne locale originale | Construite sur le DAG, comparée à 2η | 90 fois plus fine que 2η à 20 000 mains, encore 9 fois trop large | `docs/rapport_scientifique.md` §5.6 | Fait |
| Cohérence des hyperparamètres | ε_min = 1 pour l'hybride TUNED | Tableau des paramètres et explication hors politique | `docs/modele2.tex` §7, README §6 | Fait |

## État de l'art (§4)

| Demande | Vérification | Modification | Fichier ou résultat | Statut |
|---|---|---|---|---|
| Recherche réelle, sources primaires | 16 références vérifiées (DOI ou notice) ; DOI douteux non reproduits | Tableau comparatif travail / idée / recouvrement / différence / apport | `docs/etat_de_l_art.md` | Fait |
| Intégration aux documents | Bibliographies des deux `.tex` complétées | Citations dans les preuves et la discussion | `docs/modele1.tex`, `docs/modele2.tex` | Fait |
| Contribution défendable, absence de revendication excessive | Combinaison supervisé + RL non nouvelle | Contribution méthodologique et expérimentale énoncée | README §1, rapport §6 | Fait |

## Expériences (§5)

| Demande | Vérification | Modification | Fichier ou résultat | Statut |
|---|---|---|---|---|
| Protocole fixé avant les campagnes, sélection sur validation | Pilote sur graines de validation, dimensionnement 28 grappes | Protocole, addendum, configuration figée, déviations | `docs/protocole_v2*.md`, `experiments/v2/final_frozen.json` | Fait |
| E1 réplication stricte | 240 entraînements historiques et modèle 1 relancés ; écart maximal 0,0 sur gains, courbes et probabilités | Script de réplication consignant les écarts | `results/v2/replication/comparison.json` | Fait (réglage de 108 runs et notebook non réexécutés) |
| E1 baselines et oracle Monte-Carlo | 5 politiques × 100 000 parties ; |z| ≤ 1,56 | Second oracle indépendant | `results/v2/diagnostics/baselines.csv` | Fait |
| E1 équité des réglages | Bras standard à ε_min = 1 ajouté | Écart standard réglé / commun de 0,00014 seulement | rapport §5.6 | Fait |
| E2 regret contre épisodes, seuils | 12 budgets ; censure à 200 000 sans réussite fictive | Temps d'atteinte, rechutes conservées | `results/v2/analysis/threshold_runs.csv` | Fait |
| E3 coût de bout en bout | Postes chronométrés séparément ; cartes physiques comptées | Quatre conventions de coût ; contrastes à coût égal | rapport §5.5, `stage_costs.csv`, `cost_contrasts.json` | Fait (énergie non mesurée : aucun compteur fiable) |
| E4 surface budget supervisé × RL | 4 tailles × 5 budgets × 3 estimateurs, 9 grappes indépendantes | Régions de gain et de perte identifiées | `surface.json`, `budget_surface.png` | Fait |
| E5 ablations mécanistiques | Estimateurs, bruit contrôlé, exploration, ω, échantillonnage direct | Identité directe = fréquences vérifiée (2,2·10⁻¹⁶) | rapport §5.6 | Fait (variante « STAND ajustable » non faite : hors de la question étudiée) |
| E6 robustesse | <!-- E6-STATUT --> | Variante à cartes uniformes avec oracle adapté | `results/v2/variant/` | <!-- E6-FAIT --> |
| E7 statistique | Unité = jeu supervisé ; IC Student, bootstrap des grappes, TOST, effets standardisés | Multiplicité signalée, contrastes secondaires descriptifs | `src/audit/analyze.py`, `results/v2/analysis/summary.json` | Fait |

## Code, reproductibilité et documentation (§6–7)

| Demande | Vérification | Modification | Fichier ou résultat | Statut |
|---|---|---|---|---|
| Transitions et récompenses contre Gymnasium | Toutes les transitions HIT atteignables, égalités, naturels non payés | Tests ajoutés | `tests/test_audit.py` | Fait |
| Oracle indépendant, absence de fuite | Oracle d'audit sans import de l'oracle historique ; seul le bras « exact » reçoit la loi vraie | Identité à 10⁻¹⁴ près | `src/audit/oracle.py` | Fait |
| Instrumentation sans changer les trajectoires | Test d'égalité des trajectoires avec le Q-learning historique | `src/audit/core.py` | `tests/test_audit.py` | Fait |
| Intégrité des artefacts | Recalcul de J pour 418 agents ; empreintes | `scripts/verify_audit.py` | CI | Fait |
| Bug d'analyse | Clé `n` écrasée (taille de jeu remplacée par nombre de grappes) | Corrigé avant l'analyse définitive | `src/audit/analyze.py` | Fait |
| CI Linux et Windows, LaTeX | Workflow existant étendu | `verify_audit` et compilation LaTeX en mode contrôle | `.github/workflows/tests.yml` | Fait (non exécutée localement : pas d'accès GitHub Actions) |
| Licence et contribution | Absence de licence confirmée | `CONTRIBUTING.md` ajouté ; licence laissée aux auteurs | `CONTRIBUTING.md` | Fait / décision des auteurs |
| README, rapport, reproductibilité, changelog | Ancien README contenant des affirmations invalidées (25/n, « pas de surapprentissage », mécanismes garantis) | Restructuration et corrections | `README.md`, `docs/*.md`, `CHANGELOG.md` | Fait |
| PDF recompilés et vérifiés | Compilation en deux passes, contrôle des avertissements, relecture visuelle | `scripts/build_docs.py` | `docs/modele1.pdf`, `docs/modele2.pdf` | Fait |

## Demandes non satisfaites ou limitées

- **Énergie** : aucun compteur fiable n'était disponible ; aucune valeur n'est rapportée.
- **Variante « STAND préestimée mais ajustable »** : non réalisée, car elle changerait
  l'hypothèse étudiée ; l'identité de l'échantillonnage direct a été jugée prioritaire.
- **Environnement réellement différent (paquet fini)** : non réalisé ; la variante à
  cartes uniformes est une robustesse locale, pas une généralisation.
- **Comparaison des secondes entre machines** : non faite ; les durées sont propres à
  l'environnement de référence.
