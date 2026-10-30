# Contribuer et préserver la reproductibilité

Créer une branche depuis la version examinée, décrire la question scientifique
avant les expériences, et conserver configurations, graines et empreintes.
Ne pas régler un modèle sur les scores de test déjà publiés : utiliser une nouvelle
validation puis de nouvelles graines finales. Distinguer budget RL, données du
croupier, tirages de cartes, temps d'acquisition et temps d'évaluation.

Les quatre modules historiques de Wassim sont protégés par un manifeste :
`src/environment.py`, `src/stand_value.py`, `src/q_learning.py`, `src/evaluate.py`.
Toute évolution doit conserver leur version d'origine et identifier les changements
de comportement. Les variantes de recherche peuvent être ajoutées dans un module
distinct, comme `src/audit/`.

Avant une PR : `uv sync --locked`, `uv run --locked pytest -q`,
`uv run --locked python -m scripts.verify_release` et, pour l'audit,
`uv run --locked python -m scripts.verify_audit`. Une modification scientifique doit
mettre à jour preuves, résultats et interprétation ensemble. Les expériences longues
ne s'exécutent pas implicitement pendant les tests rapides.

La publication, le choix de licence et l'attribution des contributions restent des
décisions des auteurs. Aucune licence d'utilisation ou de redistribution n'est
ajoutée par cet audit ; le caractère visible du dépôt n'en fournit pas une.
