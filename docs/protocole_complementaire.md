# Protocole des expériences complémentaires

Ce protocole est écrit avant l'exécution des deux nouvelles campagnes. Les résultats
historiques du projet sont déjà connus ; il ne s'agit pas d'une préinscription externe.
Le modèle livré reste la régression C=0.1, train 60 000 mains, graine 42, associée
à l'hybride TUNED de graine 0. Les expériences ci-dessous servent à documenter
sa qualité et ses limites ; elles ne servent pas à sélectionner une meilleure graine.

## 1. Taille des données et généralisation

- Tailles train : 5 000, 20 000, 60 000 et 200 000 mains du croupier.
- Cinq jeux indépendants : répétition i=0..4, graines train=10000+3i,
  validation=10001+3i et test=10002+3i.
- À l'intérieur d'une répétition, les tailles sont des préfixes du même jeu train.
  Les tailles ne sont donc pas des observations indépendantes.
- Validation : 12 000 mains. Test : 24 000 mains, généré après le choix de C.
- C parmi 0.01, 0.1, 1, 10, 100, minimum de log-loss validation ; égalité : premier C.
  Aucun réentraînement sur validation, aucun choix basé sur la loi analytique.
- Comparaison avec les fréquences non lissées calculées sur le même train.
- Mesures : log-loss train/validation/test, Brier, accuracy, F1 macro, excès exact
  de log-loss (KL pondérée), erreur maximale des probabilités et de V_STAND.
- Moyenne, écart-type et IC approximatif de Student entre jeux, à taille fixée.
  Avec cinq jeux, les intervalles restent fragiles ; les mesures brutes sont conservées.
- Les matrices prédites, graines, empreintes de données et tous les scores de sélection
  sont exportés. Cette étude ne propage pas la variabilité supervisée dans le RL.

## 2. Ablation de l'estimateur de probabilités dans le RL

- Prédicteur livré figé et fréquences issues exactement de ses 60 000 mains train.
- Quatre méthodes : hybride logistique, hybride fréquences, hybride exact, Q-learning standard.
- Paramètres TUNED de Wassim fixés : eps_min=1 pour les hybrides, 0.4 pour le standard ;
  omega=1, gamma=1, 100 000 épisodes par entraînement.
- Vingt graines nouvelles 400..419, communes aux quatre méthodes (80 entraînements).
- Critère : gain exact J de la politique gloutonne, sans exploration à l'évaluation.
- Contraste principal : J(logistique) − J(fréquences), apparié par graine.
  Comparaisons à l'hybride exact et au standard secondaires et descriptives.
- Moyenne, écart-type et IC de Student approximatif des différences appariées ;
  pas de déclaration d'équivalence, pas de sélection à partir de ces résultats.
- Les graines communes ne garantissent pas des trajectoires identiques entre toutes
  les méthodes. La variabilité des jeux supervisés n'est pas mesurée dans cette campagne.
- Budget RL égal, pas budget total égal : coût des données et du réglage à discuter séparément.

## Décision de livraison

Conserver l'agent intégré déjà évalué, afin de livrer un point de référence reproductible.
Rapporter les limites même si une référence plus simple fait aussi bien ou mieux.
Un changement ultérieur de modèle devra être motivé par un nouveau protocole de
sélection sur validation, puis entraîner une nouvelle Q-table et une évaluation indépendante.
