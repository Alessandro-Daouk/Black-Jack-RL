> Archive historique du 28 octobre 2026, antérieure à l’intégration. Les chemins et consignes ci-dessous ne décrivent plus l’état courant. Voir ../../README.md et ../rapport.md.

# Passation : code RL de Wassim et agent hybride

Cette note permet de reprendre le travail depuis **ce dossier local**. Le dépôt
GitHub seul ne contient pas encore notre modèle supervisé et notre agent hybride.
Le dossier local n'est pas un checkout Git (il n'a pas de `.git`). Travailler ici
pour accéder aux deux contributions assemblées, puis décider séparément comment
les verser sur une branche Git propre.

## État vérifié le 2026-10-28

- Dépôt distant : `https://github.com/Alessandro-Daouk/Black-Jack-RL.git`.
- Branche `wassim` : commit `58f5ce35880ae18c107187f4b64f23f2fa005ded`, vérifié par
  `git ls-remote`. Son parent direct est `0780a4f32335540c5b59a602f38c08de3b3f01f6` :
  l'historique est aujourd'hui linéaire. Les contenus ont tout de même été comparés.
- `main` et `Alessandro-Modèle-1` pointent sur `5a9f5ab` (commit initial). Revérifier
  leur état avant toute publication.
- Les modules RL locaux de Wassim proviennent de `58f5ce3`. Seul `src/q_learning.py`
  a changé depuis `0780a4f` ; empreintes et fichiers non importés dans
  `docs/wassim_source.json`.
- `uv run --locked pytest -q` : 89 tests passent.

## Travail réalisé

1. Import sans modification de `src/q_learning.py` (`omega`, pas `1 / N ** omega`,
   `TUNED`). `src/tune.py` n'est pas importé : il dépend de `pandas` et règle les
   paramètres avec la V_STAND exacte. `pyproject.toml` et `uv.lock` sont inchangés.
2. `src/train_hybrid.py` exige `--preset config` ou `--preset tuned` (`CONFIG` ou
   `TUNED["hybrid"]`). L'archive et `training.json` enregistrent `training_preset` et
   la configuration complète, omega compris. gamma=1 et V_STAND figée sont vérifiés.
3. Les sorties par défaut de l'entraînement et de l'évaluation suivent le nom de
   l'agent : `results/hybrid/<nom>/`. L'agent par défaut est
   `models/hybrid/agent_tuned.npz`.
4. `src/compare_hybrid.py` compare CONFIG et TUNED pour l'hybride supervisé, à
   100 000 épisodes. Le choix se fait sur les graines de validation 200–219, le test
   sur les graines 0–19, avec le standard et l'hybride exact comme références.
   Résultats : `results/hybrid/config_vs_tuned/`.
5. Agent retenu : TUNED, graine 0, évalué sur 10 000 parties de graine 2027 et par
   le calcul exact. Résultats : `results/hybrid/agent_tuned/`.
6. L'ancien agent `models/hybrid/agent.npz` et ses résultats à la racine de
   `results/hybrid/` sont inchangés octet pour octet. `results/supervised/integration.json`
   (vérification de `0780a4f`) est conservé ; celle de `58f5ce3` est dans
   `results/supervised/integration_58f5ce3.json`.

Vérifications de cohérence : avec `CONFIG` et la graine 0, le nouveau code reproduit
l'ancien agent bit à bit. Nos entraînements TUNED du standard et de l'hybride exact
retrouvent exactement les gains de `results/metrics.csv` de Wassim pour les graines 0–4.

## Résultats à retenir

- Validation : TUNED meilleur sur 20 graines sur 20 ; test : 17 sur 20, +0.000737 de
  gain exact moyen (p exact = 0.0019). L'écart moyen à l'optimum passe de 0.001528 à 0.000791.
- La politique limite de notre V_STAND estimée ne perd que 0.000022 face à l'optimum ;
  avec TUNED, l'hybride supervisé ne se distingue pas de l'hybride exact (p = 0.77).
- L'agent par défaut a un gain exact de -0.047207, contre -0.047765 pour l'ancien
  et -0.046556 pour l'optimum.
- Face au standard, la conclusion dépend du budget compté pour le prédicteur ;
  voir le README et le rapport de comparaison.

## Suite possible

- Mesurer la variabilité due aux données du prédicteur (autres graines ou tailles
  d'entraînement supervisé), puis étudier d'autres budgets RL.
- Ne plus faire de choix sur les graines de test 0–19, ni sur les parties
  d'évaluation 2026 et 2027 : utiliser de nouvelles graines.
- Préparer la branche Git : réunir README, `pyproject.toml` et `uv.lock` avec ceux de
  Wassim (le sien ajoute `pandas` et `ipykernel`), puis régénérer le verrou après accord.

## Environnement et commandes de départ

```bash
export PATH="$PWD/.tools/uv/bin:$PATH"
uv sync --locked
uv run --locked pytest -q
uv run --locked python -m src.play --state 15 10 0
```

La documentation du projet est dans `README.md` et `docs/agent_hybride.md` ; les
résultats dans `results/supervised/report.md`, `results/hybrid/config_vs_tuned/report.md`
et `results/hybrid/agent_tuned/report.md`.
