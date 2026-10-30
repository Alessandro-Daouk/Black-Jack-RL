# Audit initial — 30 octobre 2026

Référence ouverte : `9c3d6318fe3d499e4f8e50b5680df1aabf459ff8`, branche
`main`, arbre propre. Branche de travail : `audit/scientifique-v2`.
Ce constat précède les modifications scientifiques et les nouvelles campagnes.
L'environnement détaillé et les empreintes des fichiers initiaux sont dans
[`results/v2/baseline/environment.json`](../results/v2/baseline/environment.json).

## Vérifications effectuées

Installation neuve avec le `uv.lock` existant : Python 3.12.7, NumPy 2.5.4,
scikit-learn 1.9.1, SciPy 1.18.1, Gymnasium 1.4.0. Linux x86_64,
Intel i7-1255U, 12 processeurs logiques, environ 30 Gio de RAM.
Les 98 tests passent (12,74 s). `scripts.verify_release` passe : sept artefacts
intègres, J livré = −0,047206652727067894, J optimal = −0,046555977066697535.
L'accès réseau initial était interdit par le bac à sable ; l'installation a réussi
après autorisation. La création de branche a également nécessité l'accès à `.git`.

Lecture intégrale : README, deux sources TeX, protocole complémentaire, modules
de simulation, oracles, apprentissage, sélection, comparaisons, démonstration,
vérification et tests. Inspection supplémentaire du source **installé** de
`LogisticRegression`, `LinearModelLoss` et `BlackjackEnv`.

## Inventaire et traçabilité

| Hypothèse ou affirmation | Code / preuve initiale | Mesure disponible | Diagnostic initial |
|---|---|---|---|
| Loi du croupier conditionnée seulement par sa carte visible | `src/supervised/data.py`, `reference.py`, `stand_value.py` | Tests contre Gym, probabilités exactes | Correct dans le paquet infini ; renforcer par oracle indépendant |
| Régression saturée proche des fréquences | `model.py`, `modele1.tex` | 60 probabilités et cinq réplications | Objectif correct ; précision sur existence et invariance du biais requise |
| Espérance du KL brut = environ 25/n | `modele1.tex`, README §5.7 | Cinq réalisations sans zéro | **Faux en espérance non conditionnelle à n fini** : un zéro a probabilité positive |
| KL exact calculé | `supervised/evaluate.py:metrics` | `exact_excess_log_loss` | Écrêtage à epsilon machine : pas le KL brut si une case est nulle |
| Absence de surapprentissage prouvée | README §5.7, conclusion M1 | Faibles écarts train/test | Conclusion excessive ; petit modèle bien spécifié ≠ absence de variance/optimisme de sélection |
| DAG, au plus 18 décisions | `modele2.tex`, transitions | Tests partiels | **Borne erronée** : deux as donnent un total dur initial de 2 ; borne correcte 20 |
| Convergence vers l'optimum véritable | M2 §4–5 | Courbes finies | Vraie seulement pour STAND exacte ; distinguer optimum du modèle estimé et hypothèses stochastiques |
| Réduction garantie du biais et de la variance | M2 §5, README §7.2 | Ablation globale | Mécanisme plausible ; pas de domination universelle des cibles |
| Borne regret 36η | M2 §6 | Erreur limite de l'agent | Facteur de durée erroné et inutile : une seule récompense terminale perturbée permet 2η |
| Hybride meilleur à budget RL égal | `compare_hybrid.py`, `compare_frequencies.py` | 20 graines, un prédicteur fixé | Répliquer ; distinguer réglages propres et identiques, propager les graines supervisées |
| Gain de coût global | README §9.5 | Addition mains / parties | Unités non équivalentes ; aucun gain temporel total établi |
| Temps d'entraînement | `q_learning.py:train` | `training_seconds` | Inclut les évaluations exactes ; isoler les postes en v2 |
| Oracle réservé à l'évaluation | Comparaisons / entraînement | Sources | Vrai pour supervisé et standard ; **exception explicite** : hybride exact reçoit STAND vraie |
| Intégrité de Wassim | `docs/wassim_source.json` | Empreintes vérifiées | Préserver les quatre modules et leur manifeste |

Les politiques ne sont pas invalidées par les erreurs de preuve identifiées. Aucun
défaut de transition n'est établi à ce stade. La réplication, l'équité et le coût
restent des questions expérimentales, pas des résultats présupposés.

## Priorités et plan

1. Corriger les théorèmes (KL, identifiabilité, DAG, convergence, regret) et qualifier
   les mécanismes ; sourcer l'état de l'art sans revendiquer un nouvel algorithme.
2. Ajouter un oracle indépendant et des tests exhaustifs des transitions atteignables,
   des comptes de cartes, des bornes et de l'identité de différence de performance.
3. Fixer le protocole pilote puis calculer le nombre de grappes indépendantes sur
   validation ; figer les graines finales avant les nouvelles mesures.
4. Répliquer les campagnes historiques principales ; mesurer E1–E5 avec coûts réels,
   incertitude par jeu supervisé et censure des temps d'atteinte. Tester une variante
   de loi de cartes si ses contrôles sont satisfaits.
5. Produire résultats v2, rapport, réponse à la revue, documentation de reproduction,
   PDF compilés et contrôlés, puis tests/release finaux.

Les sorties historiques, modèles et protocole complémentaire restent immuables.
La licence, absente, ne sera pas choisie à la place des auteurs. La branche locale
permet de réviser les modifications avant publication.
