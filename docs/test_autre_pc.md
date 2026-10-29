# Valider le dépôt sur un autre PC

Objectif : vérifier que le dépôt est autonome, sans récupérer `.venv`, `.tools`
ou des fichiers de l'ordinateur d'origine. Installer Git et uv avec leurs
installateurs officiels, puis ouvrir un nouveau terminal si nécessaire.

```bash
git clone --branch integration/supervised-hybrid https://github.com/Alessandro-Daouk/Black-Jack-RL.git
cd Black-Jack-RL
git branch --show-current
git rev-parse HEAD
uv --version
uv sync --locked
uv run --locked python --version
uv run --locked python -m scripts.verify_release
uv run --locked pytest -q
uv run --locked python -m src.play --state 15 10 0
uv run --locked python -m src.play --games 5 --seed 2030 --trace
```

Résultats attendus :

- branche `integration/supervised-hybrid` ; noter le commit affiché ;
- Python 3.12.7 et aucune modification de `uv.lock` ;
- vérification `status: passed`, gain exact environ `-0.047206652727` ;
- tous les tests passent ; le nombre peut évoluer avec les commits ;
- état `(15, 10, 0)` : HIT, STAND ≈ −0,582872 et HIT ≈ −0,541728 ;
- cinq parties terminées et des décisions affichées ; gagner chaque partie n'est pas un critère de validation.

La première installation demande Internet. Les démonstrations et les tests sont
ensuite locaux. Sous Windows, ces commandes fonctionnent aussi dans PowerShell ;
aucune activation manuelle de `.venv` n'est nécessaire.

Pour tester le notebook de Wassim : `uv sync --locked --extra notebook`, ouvrir
`train_model2.ipynb` dans un éditeur de notebooks, sélectionner le Python de `.venv`.
Exécuter le notebook relance des entraînements et réécrit ses résultats historiques :
faire cet essai dans une copie de travail dédiée.

Pour signaler un problème, conserver le système d'exploitation, le commit,
`uv --version`, la commande exacte et son message d'erreur complet.
Après réussite de ce test, la suite prévue est la revue/fusion de la branche,
la création d'une version de rendu, puis l'application web.
