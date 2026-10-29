# Intégration, provenance et contributions

## Historique conservé

La branche `integration/supervised-hybrid` part de `wassim`, commit
`58f5ce35880ae18c107187f4b64f23f2fa005ded`. Elle conserve les commits antérieurs
`0780a4f` et `5a9f5ab`, plutôt que recréer un historique à partir de copies isolées.
La branche main et les branches personnelles peuvent continuer à évoluer séparément
jusqu'à la revue et à la fusion de l'intégration.

Les modules `src/environment.py`, `src/stand_value.py`, `src/q_learning.py`,
`src/evaluate.py`, les tests RL, `src/tune.py`, le notebook, les résultats
historiques et le rapport original de Wassim sont conservés. Les empreintes des
modules et tests utilisés par l'hybride sont dans `wassim_source.json`.

## Contributions assemblées

- Wassim : environnement et références exactes, Q-learning standard/hybride,
  réglage eps_min/omega, notebook, premières comparaisons et ablations.
- Alessandro : simulation supervisée, classificateur et baseline par fréquences,
  évaluation probabiliste, interface de l'agent intégré, campagnes complémentaires
  et documentation commune. Ce travail de développement a été assisté par des agents IA.
- L'intégration conserve les origines des fichiers ; les noms dans les commits
  historiques restent ceux de leurs auteurs.

Le projet initial est conservé dans `blackjack_ai_projet_modele_1.md` comme document
de contexte. La passation antérieure est archivée dans `docs/archive/` ; elle n'est
pas un guide de l'état actuel. `docs/rapport.md` rassemble la conclusion commune.

## Réconciliation technique

- Dépendances communes dans `pyproject.toml` et `uv.lock`, Python 3.12.7.
- pandas est inclus pour le réglage et le notebook de Wassim ; le noyau ipykernel
  est proposé via l'extra `notebook`.
- Tests originaux conservés dans `tests/`, nouveaux tests supervisés dans
  `tests/supervised/` ; aucune copie dupliquée des tests RL.
- Fichiers texte UTF-8, fins de ligne LF pour les sources et manifestes, afin de
  conserver les empreintes sous Windows comme sous Linux.
- Toute lecture ou écriture de fichier texte précise `encoding="utf-8"` : sous Windows,
  l'encodage par défaut est cp1252. La CI ne force pas l'UTF-8 et fait échouer pytest
  sur tout fichier ouvert sans encodage explicite (`PYTHONWARNDEFAULTENCODING=1`).
- Les environnements, caches et reproductions locales sont exclus de Git.
- Vérification des modèles livrés : `python -m scripts.verify_release`.
- Workflow GitHub Actions préparé pour Linux et Windows ; son résultat effectif
  dépend de son exécution sur GitHub, distincte des contrôles locaux.

## Corrections issues de la review

1. L'entraînement hybride exige une sortie explicite et refuse les collisions.
   Un remplacement autorisé par `--overwrite` invalide les anciens rapports d'évaluation.
2. L'hybride avec probabilités exactes est décrit comme référence, pas comme borne
   de performance à nombre fini d'épisodes. La borne est la politique optimale.
3. Les hypothèses des permutations de signes et les limites des IC sont explicites ;
   une absence de différence n'est pas présentée comme une preuve d'équivalence.
4. L'ablation ajoute les fréquences du même train que la régression ; les courbes
   supervisées ajoutent plusieurs jeux indépendants et tailles de données.

Les anciens CSV, JSON et modèles restent inchangés ; les rapports générés ont été
clarifiés. Les nouvelles mesures sont rangées séparément, avec leur propre protocole.
Les empreintes de sources dans les résultats décrivent leur campagne ; elles peuvent
différer après des corrections de documentation ou de portabilité des exports, sans
changement des calculs ni réécriture des mesures. Le manifeste des modèles livrés,
lui, doit correspondre exactement aux artefacts du dépôt.

## Prochain jalon

Tester depuis un clone sur un autre PC avec `docs/test_autre_pc.md`. Après ce contrôle,
ouvrir/revoir la PR vers main, fusionner puis créer une version de rendu.
L'application web viendra ensuite et utilisera l'interface `HybridAgent`, avec les
mêmes règles et les modèles déjà chargés. Aucun service web n'est inclus à ce stade.
