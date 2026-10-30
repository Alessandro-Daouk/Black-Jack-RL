# Protocole v2 — fixé avant les nouvelles campagnes, 30 octobre 2026

Les résultats historiques sont connus. Ceci est un protocole interne daté, pas une
préinscription externe. La livraison historique reste inchangée. Aucun résultat du
test v2 ne sert à choisir un modèle livré ou ses hyperparamètres.

## Question, critères et précision

Question : quand fournir une estimation figée de STAND aide-t-il, en épisodes RL,
en tirages de cartes et en secondes, coût d'acquisition inclus ou amorti ?
Critère principal : différence appariée de J, logistique moins standard TUNED,
à 20 000 mains supervisées et 100 000 épisodes RL. Différence pratique minimale
fixée à **0,001 récompense par partie** (0,1 mise pour 100 parties).
Précision visée : demi-largeur de l'IC à 95 % de **0,0005** sur ce contraste.
Les autres tailles, méthodes, budgets, temps et variantes sont secondaires et
descriptifs : pas de balayage de p-values ni de sélection de la meilleure graine.
Pas de déclaration d'équivalence par absence de significativité.

## Pilote de validation et règle de dimensionnement

Quatre jeux train indépendants (graines 31000, 31010, 31020, 31030), validations
31001, 31011, 31021, 31031 (6 000 mains), deux graines RL par jeu
`32000 + 10*i + j`, j=0,1. Train 20 000 mains ; 100 000 épisodes.
Comparer logistique (C sélectionné sur validation parmi .01,.1,1,10,100) et standard
TUNED. Les quatre différences moyennées sur j sont les quatre unités indépendantes.
Estimer leur écart-type s. Le nombre final G est le plus petit entier ≥8 tel que
`t(.975,G-1) * (1.5*s) / sqrt(G) <= .0005`. Le facteur 1,5 est une précaution
contre l'imprécision de quatre grappes pilotes, pas une garantie de puissance.
Consigner s, G et la demi-largeur attendue avant le lancement final. Aucun arrêt
anticipé selon les scores finaux. Si la précision réalisée diffère, la rapporter.

## E1 — réplication et contrôles

Réexécuter les 160 runs CONFIG/TUNED et les 80 runs de l'ablation, avec les modules
historiques dont les empreintes restent celles du commit de référence. Comparer
les J et courbes aux CSV publiés (tolérance 1e-9 pour courbes arrondies). Réentraîner
le prédicteur livré avec les graines historiques, comparer ses probabilités et J.
Ce sont des réplications, jamais des données de test neuves.

Oracle indépendant : propagation de masse du croupier, matrice de transition HIT,
évaluation des politiques par système linéaire ou parcours topologique indépendant.
Contrôler masses, récompenses, toutes les transitions atteignables avec Gym.
Baselines définies : STAND partout, HIT partout, random Bernoulli(.5) à chaque
décision, règle tabulaire `HIT si total <17`, politique optimale. Monte-Carlo
100 000 parties chacune, graines 35000..35004 ; IC normal 95 %, erreur Monte-Carlo
distincte de celle d'entraînement. Un dépassement d'IC isolé n'est pas un bug établi.

## E2–E4 — plan imbriqué et coût complet

G jeux supervisés indépendants déterminés par le pilote, **deux** réplications RL
par jeu (ce nombre fixe répartit le calcul entre les deux sources de variabilité).
Tailles train **1 000, 5 000, 20 000, 60 000**, préfixes d'un même jeu au sein d'une
grappe. Train `41000+10*i`, validation `41001+10*i`, test `41002+10*i`,
6 000 mains validation et 12 000 test. RL `51000+10*i+j`.
C choisi sur la seule validation ; pas de refit. Test généré après tous les choix.

Méthodes : logistique, fréquences brutes, Jeffreys (alpha=1/2), hybride exact
(oracle privilégié), standard TUNED (epsilon minimum .4), standard commun
(epsilon minimum 1). Hybrides epsilon minimum 1, omega=1 ; gamma=1 ; decay=.99995.
Standard/oracle entraînés une seule fois par graine, réutilisés dans les comparaisons
de tailles, jamais comptés comme de nouvelles observations indépendantes.

Points d'observation : 0,100,500,1 000,2 000,5 000,10 000,20 000,40 000,60 000,
100 000,200 000 épisodes. Seuils de regret : .01, .005, .002, .001.
Temps d'atteinte = premier point observé sous le seuil ; les non-atteintes sont
censurées à droite à 200 000, **pas remplacées par une réussite à 200 000**.
Rapporter proportion de réussites, médiane conditionnelle et coût du premier
franchissement ; une rechute ultérieure reste possible. La grille limite la résolution.

Gymnasium historique conservé pour les entraînements. Compter les cartes réellement
distribuées (joueur+croupier, y compris cachée et bust) par les tailles des mains.
Les tirages cosmétiques de couleurs de Gym ne sont pas des cartes de jeu.
Temps RL exclut les évaluations exactes ; ces dernières sont enregistrées séparément.
Séparer simulation train/validation/test, fits candidats, fit retenu, prédiction,
fréquences/lissage, construction STAND, chargement, RL, évaluations. Temps muraux,
ordre des méthodes mélangé de façon déterministe, warm-up, un processus et un thread
BLAS/OpenMP. Aucun entraînement concurrent pendant les mesures principales.
Ordinateur portable non isolé : conserver dispersion et limites des secondes mesurées.

Conventions : (a) production avec C déjà fixé : train + fit retenu + STAND + RL ;
(b) acquisition avec sélection : train + validation + tous les fits/scores + STAND
+ RL ; (c) validation scientifique : ajouter test, métriques et évaluations ;
(d) réutilisation : chargement de la table + RL, ou amortissement acquisition/M.
Le coût du tuning RL historique n'est pas inventé : les mesures historiques ne
l'isolent pas. Rapporter son budget (108 runs de 100 000 épisodes) séparément et
mesurer les variantes v2 comme coût de développement, pas coût obligatoire de production.
Mémoire : taille tableaux et pic RSS du processus (inclut imports et travaux précédents),
sans l'interpréter comme une allocation propre à chaque méthode. Pas de mesure d'énergie
en l'absence de compteur fiable. Ne jamais assimiler une main croupier à une partie.

## E5 — mécanismes et statistiques catégorielles

Probabilités perturbées valides `(1-rho)*P + rho*delta_BUST`, rho=0,.01,.05,.20 ;
sensibilité opposée vers delta_21. Examiner limite DP, erreurs, marges et regret.
RL : rho=.05 et .20 vers BUST, omega=.8 et epsilon minimum .05 sur l'hybride
logistique à 20 000 mains ; comparaisons avec le bras principal, paramètres fixes.
Pas de tuning sur ces scores. Échantillonnage direct de STAND : moyenne de R(p,D)
sur les mêmes mains ; vérifier son identité avec `frequences @ recompenses`.
Variante ajustable non indispensable : l'identité du contrôle direct est prioritaire.

Validation de 25/n : tirages multinomiaux directs de la loi jointe (instrument
mathématique utilisant l'oracle, **pas données d'entraînement d'un agent**),
n=50,100,500,1 000,5 000,20 000,60 000,200 000 ; 2 000 répétitions par taille,
graine 61000. Fréquences brutes : compter les KL infinis, ne pas les écrêter ni
annoncer une espérance finie en l'absence de zéros observés. Jeffreys et Laplace :
moyenne, IC, produit n*KL. Pour une carte à n_d fixé, vérifier la dispersion du
terme quadratique plutôt que la remplacer par sa moyenne.

## E6 — robustesse optionnelle contrôlée

Variante à cartes uniformes sur 1..10 (au lieu de quatre dix sur treize), mêmes
règles et état. Oracle adapté, simulateur vérifié contre la loi et les transitions.
Comparer standard et hybride fréquence/Jeffreys à 20 000 mains avec nouvelles
graines 71000+10*i (données), 81000+10*i+j (RL). Ni transfert sans réentraînement
ni généralisation aux casinos ; seulement robustesse locale à une loi différente.

## E7 — unités, intervalles et traçabilité

Calculer d'abord les différences appariées entre méthodes à graine commune, puis
moyenner les deux graines RL **dans chaque jeu supervisé**. IC de Student sur G
moyennes indépendantes, complété par bootstrap des grappes entières (10 000 tirages,
graine 91000) et médiane / étendue pour sensibilité aux extrêmes. Ce bootstrap
préserve l'imbrication ; il n'augmente pas artificiellement le nombre de jeux.
IC marginaux descriptifs non simultanés hors contraste principal. Courbes : bande
de dispersion entre grappes explicitement distinguée d'un IC sur la moyenne.
Occupation = nombre espéré de visites par épisode sous la politique évaluée,
non normalisé ; vérifier l'identité regret = occupation × désavantage optimal.

Toutes les campagnes refusent une sortie existante ; configs, graines, commit,
empreintes source/protocole, versions, chronométrages et courbes sont conservés.
Les scripts d'analyse doivent régénérer les tableaux/figures depuis les mesures,
sans modifier les données. Toute déviation sera inscrite dans un fichier séparé.
