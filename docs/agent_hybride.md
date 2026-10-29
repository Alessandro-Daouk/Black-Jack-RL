# Comment les deux modèles coopèrent

Le modèle supervisé est assez proche de la référence exacte pour passer à
l'intégration : sur les 60 probabilités possibles, l'écart maximal est de 1.90
point de pourcentage. L'erreur maximale de gain espéré de STAND est de 0.03871.
Les petites erreurs peuvent encore modifier des décisions proches de l'égalité ;
c'est la performance de l'agent complet qui permet d'en mesurer l'effet pratique.

L'accuracy de 36.49 % est celle de la classe du croupier la plus probable, pas un
taux de bonnes décisions HIT/STAND. Sur ce test, le prédicteur exact obtient la
même accuracy. Les probabilités conditionnelles restent la bonne sortie pour
calculer une espérance de récompense.

## Décision pendant une partie

Pour l'état observable `(total_joueur, carte_visible, as_utilisable)` :

1. Le modèle supervisé fournit les six probabilités du résultat du croupier.
2. On calcule le gain espéré de rester : chaque probabilité est multipliée par
   la récompense correspondante, puis les résultats sont additionnés.
3. Le modèle RL fournit Q_HIT, le gain espéré de tirer puis de continuer à jouer.
4. L'agent choisit HIT si Q_HIT est strictement supérieur à V_STAND ; sinon STAND.

Le RL utilise toujours l'état du joueur, la carte visible et l'as utilisable.
La sortie supervisée fournit la valeur de l'action STAND au même état. Elle
intervient aussi dans la cible du Q-learning pendant l'apprentissage.

Avec le modèle livré et l'agent par défaut, à l'état `(15, 10, 0)` :

- V_STAND vaut environ -0.582872 ;
- Q_HIT vaut environ -0.541728 (-0.546372 pour l'ancien agent) ;
- l'agent choisit HIT, car cette valeur est plus élevée.

Les deux valeurs peuvent être négatives. L'agent cherche alors l'action dont
la perte moyenne estimée est la plus faible. Une décision n'est pas une garantie
sur le résultat d'une main.

L'agent hybride est une classe qui assemble les deux composants. Les scripts
servent à l'entraîner, jouer ou l'évaluer. Seules dix cartes visibles sont possibles :
on calcule les dix distributions et V_STAND une fois au chargement, puis on consulte
les valeurs en mémoire pendant la partie.

## Entraînement

Le prédicteur supervisé est entraîné en premier et reste ensuite figé. Pour chaque
transition de HIT, le Q-learning de Wassim apprend avec la cible :

```text
si la manche est terminée : cible = récompense
sinon : cible = récompense + max(Q_HIT(état_suivant), V_STAND(état_suivant))
```

Le facteur d'actualisation vaut 1 dans ce jeu épisodique. La valeur de STAND n'est
jamais mise à jour par le RL. Chaque Q_HIT se rapproche de sa cible avec le pas
`1 / N ** omega`, où N compte les visites de cet état ; `omega = 1` donne une moyenne
des cibles. Pendant l'apprentissage, l'agent explore aussi des actions ; pendant
la démonstration et l'évaluation, il choisit toujours la plus grande valeur.

Deux jeux de paramètres de Wassim sont disponibles, à choisir avec `--preset` :

- `config` : exploration décroissante de 1 à 0.05, la configuration précédente ;
- `tuned` : `TUNED["hybrid"]`, exploration maintenue à 1. Les actions d'apprentissage
  sont alors aléatoires, mais le Q-learning apprend quand même la politique gloutonne.

Le notebook original de Wassim sauvegardait une Q-table entraînée avec la
distribution exacte du croupier. Pour notre système, deux Q-tables ont été entraînées
sur 100 000 épisodes avec notre prédicteur, graine 0. La première utilise `config`
et le code de `0780a4f` : c'est l'ancien agent. La seconde utilise `tuned` et le code
de `58f5ce3` : c'est l'agent par défaut. Sur 20 graines de test, `tuned` divise
environ par deux l'écart moyen à l'optimum (0.000791 contre 0.001528). Le code original
d'apprentissage est conservé ; seul son paramètre V reçoit la table du modèle supervisé.

## Quand réentraîner

- Relancer le modèle supervisé avec les mêmes données et paramètres ne devrait
  pas améliorer ses probabilités. Pour étudier une amélioration, augmenter les
  simulations ou comparer plusieurs graines, avec validation et test indépendants.
- Si le prédicteur change, réentraîner le Q-learning hybride avec la nouvelle
  valeur de STAND permet d'obtenir un couple cohérent et comparable.
- Si le code RL de Wassim change, l'importer sans modification, mettre à jour
  `docs/wassim_source.json`, vérifier qu'il reproduit l'ancien agent avec les mêmes
  paramètres, puis comparer les réglages sur des graines de validation distinctes du test.
- Si seul l'affichage ou l'interface web change, les modèles sauvegardés restent
  utilisables tels quels.
- Si les règles du jeu changent, revoir le simulateur, les récompenses et les deux
  apprentissages ; nos artefacts correspondent à Blackjack-v1 avec natural=False
  et sab=False.

Le chargement vérifie l'empreinte du fichier supervisé et la table STAND pour éviter
d'associer deux versions différentes. Il conserve les nombres de visites de HIT,
ce qui permet de voir quelles valeurs reposent sur peu d'expériences.

## Fichiers et usage

- `src/hybrid_agent.py` : classe HybridAgent, décision et chargement des deux composants.
- `src/train_hybrid.py` : entraînement RL avec le prédicteur figé et le réglage choisi.
- `src/play.py` : démonstration de parties et explication d'un état.
- `src/evaluate_hybrid.py` : évaluation sans exploration sur des parties indépendantes.
- `src/compare_hybrid.py` : comparaison CONFIG / TUNED sur plusieurs graines.
- `models/supervised/dealer_model.joblib` : prédicteur supervisé.
- `models/hybrid/agent_tuned.npz` : agent par défaut (Q_HIT, visites, V_STAND, métadonnées).
- `models/hybrid/agent.npz` : ancien agent, conservé tel quel.
- `results/hybrid/agent_tuned/` : paramètres, courbe d'apprentissage et évaluation de l'agent par défaut.
- `results/hybrid/config_vs_tuned/` : comparaison des réglages, avec les références.
- Racine de `results/hybrid/` : résultats d'origine de l'ancien agent.

Depuis la racine du projet :

```bash
uv run --locked python -m src.play --games 5 --trace
uv run --locked python -m src.play --state 15 10 0
```

La comparaison sur 20 graines de test place l'hybride supervisé devant le Q-learning
standard à budget RL égal. Ce résultat dépend du coût compté pour le prédicteur. Dans
un budget total de 100 000 simulations, ses 60 000 mains d'entraînement laissent
40 000 épisodes RL, et l'hybride reste devant en moyenne. Avec les 84 000 mains simulées
pour l'entraîner, le valider et le tester, il ne reste que 16 000 épisodes, et le
standard repasse devant.
Les expériences complémentaires mesurent maintenant la variabilité du prédicteur
sur cinq jeux simulés et quatre tailles, et comparent l'hybride à une estimation
par fréquences sur le même train. Elles sont décrites dans [le rapport intégré](rapport.md).
La propagation de la variabilité supervisée dans plusieurs entraînements RL reste à étudier.

Pour entraîner une nouvelle table, fournir explicitement `--agent` et un dossier de
résultats distinct. Les sorties existantes sont refusées sans `--overwrite` ; ce
remplacement invalide leur ancienne évaluation. Voir [les commandes de reproduction](reproduction.md).
