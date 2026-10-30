# Protocole v2 — compléments après relecture (E7), 30 octobre 2026

Écrit après la campagne v2 et après une relecture critique, avant les mesures E7.
Un essai rapide hors protocole (20 graines, loi des cartes estimée sur toutes les cartes
des mains du croupier) avait déjà donné un regret d'environ 0,0001 à 1 000 mains pour la
planification complète : ces compléments sont donc **exploratoires**, pas confirmatoires.
Ils ne modifient ni l'agent livré, ni la campagne finale, ni ses conclusions publiées ;
ils les situent. Code : `src/audit/complements.py` ; sorties : `results/v2/complements/`.

## Questions

1. **Planification.** Les mains du croupier contiennent des cartes tirées dans la même
   loi que celles du joueur. Si l'on estime cette loi, on peut calculer HIT par
   programmation dynamique au lieu de l'apprendre. Que vaut cette référence face à
   l'hybride, avec les mêmes données ?
2. **Double Q-learning.** Le standard réglé est un Q-learning tabulaire simple. Un
   Double Q-learning, qui retire le biais du maximum, réduit-il l'écart avec l'hybride ?
3. **Réglage sans privilège.** Les hyperparamètres TUNED de l'hybride ont été choisis
   avec la loi exacte du croupier. Le même réglage, fait avec un modèle 1 estimé,
   choisit-il la même configuration ?
4. **Seuil de 0,001.** Donner des repères chiffrés au seuil fixé dans le protocole v2.

## Données et graines

- Jeux supervisés : ceux de la campagne finale (graines `41000 + 10*i`, i = 0..27),
  régénérés et contrôlés par l'empreinte `train_sha256` de `results/v2/final/fits/`.
  Les probabilités logistiques sont lues dans ces mêmes fichiers, sans réajustement.
- Graines RL de Double Q-learning : `51000 + 10*i + j`, j = 0,1, comme la campagne finale.
- Réglage : graines RL 100, 101, 102 et grille de `src/tune.py`
  (ε_min ∈ {0,05 ; 0,1 ; 0,2 ; 0,4 ; 0,6 ; 1}, ω ∈ {1 ; 0,8 ; 0,6}), 100 000 épisodes,
  critère : regret final moyen sur les trois graines. Pour la question 3, chaque graine
  reçoit un nouveau jeu supervisé de 20 000 mains (graines `61000 + 10*k`, validation
  `61001 + 10*k`, k = 0,1,2) et le modèle logistique choisi sur sa validation.

## Méthodes

- **Planification, STAND estimée** : STAND donnée par le modèle logistique du jeu
  (comme l'hybride) ; transitions de HIT calculées avec la loi des cartes estimée par
  les fréquences des **cartes visibles** du jeu (n tirages indépendants, sans biais
  d'arrêt). Aucune partie jouée. Hypothèse supplémentaire par rapport à l'hybride : le
  joueur connaît l'arithmétique de son total.
- **Planification complète** : même loi des cartes estimée, loi du croupier déduite de
  la règle d'arrêt à 17. Hypothèse supplémentaire : la règle du croupier est connue.
- **Double Q-learning** (van Hasselt, 2010) : deux tables, mise à jour de l'une avec
  l'action choisie par elle et évaluée par l'autre, pas 1/N^ω, ε décroissant comme le
  standard, politique gloutonne sur la somme. Réglé comme le standard (question 3 :
  même grille, mêmes graines).

## Analyses

Unité : le jeu supervisé, moyenne des deux graines RL. Intervalles de Student à 95 %
entre jeux ; aucun test de significativité supplémentaire.

- A1 : regret des deux planifications à 20 000 mains (28 jeux) et à 1 000, 5 000,
  20 000, 60 000 mains (9 premiers jeux) ; différence appariée avec l'hybride
  logistique à 100 000 épisodes.
- A2 : configuration Double Q choisie ; différences appariées Double Q − standard et
  hybride − Double Q à 10 000, 40 000 et 100 000 épisodes (28 jeux).
- A3 : configuration choisie pour l'hybride sans STAND exacte. Règle fixée d'avance :
  si elle diffère de (ε_min = 1, ω = 1), elle est entraînée sur les 28 jeux à
  100 000 épisodes et son contraste avec le standard est rapporté ; sinon rien n'est
  relancé.
- A4 : regret de chaque politique qui ne diffère de π\* qu'en un seul état, et part de
  l'avantage de la maison (0,0466) que représente 0,001.
