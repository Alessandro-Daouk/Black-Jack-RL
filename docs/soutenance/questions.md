# Soutenance : déroulé, démonstration et questions probables

Diapositives : [diapositives.html](diapositives.html) (flèches pour naviguer, N pour les notes,
F pour le plein écran). Durée visée : 12 minutes de présentation, 3 de démonstration.

## Déroulé

| Diapositives | Message | Durée |
|---|---|---:|
| 1–3 | La question et la méthode : estimer STAND à part, n'apprendre que HIT | 2 min |
| 4 | L'oracle exact : sans lui, les écarts du millième sont invisibles | 1 min |
| 5–6 | Le résultat principal et la comparaison à six algorithmes | 3 min |
| 7–8 | Les limites : le plancher du modèle, les résultats négatifs | 2 min |
| 9–11 | Le sabot fini, la référence à information égale, la confirmation | 3 min |
| 12 | Conclusion : la séparation accélère, l'information fixe le niveau | 1 min |

## Démonstration (3 minutes)

1. Ouvrir `web/index.html`. Appuyer sur Espace pour distribuer, puis C pour afficher les conseils :
   montrer que les trois modèles donnent leur décision et leurs valeurs STAND et HIT.
2. Jouer une main contre le conseil de l'optimal, puis montrer à droite que les modèles ont joué la
   même donne avec les mêmes cartes, et le message « décisions conformes ».
3. Passer en « Sabot fini », jouer deux donnes : montrer le compte vrai qui évolue et un état où
   l'hybride linéaire et la stratégie de base ne décident pas pareil.
4. Conclure sur les scores : sur quelques donnes, la chance domine ; c'est pourquoi le projet
   mesure des gains exacts.

## Questions probables

**Pourquoi ne pas simplement simuler beaucoup de parties pour évaluer les agents ?**
Le gain d'une partie a un écart-type de 0,95 : mesurer un écart de 0,001 demanderait environ
3,5 millions de parties par politique. L'oracle exact donne le gain sans bruit, et il est
lui-même contrôlé par un second oracle écrit indépendamment, par Gymnasium et par Monte-Carlo.

**L'hybride reçoit de l'information gratuite : la comparaison est-elle équitable ?**
Non, pas gratuite : le coût des mains de croupier est mesuré (§5.5). Une main coûte environ un
tiers de partie ; au même temps total, acquisition comprise, l'hybride reste devant entre 2 et
20 s. Il perd quand l'acquisition consomme le budget ou quand le modèle est trop imprécis.

**La régression logistique sert-elle à quelque chose ?**
En paquet infini, non : avec une seule variable catégorielle, elle reproduit les fréquences
(équivalence établie à ±0,001). En sabot fini, oui : sur la composition du sabot, elle bat la
table de 0,0143 nat de KL et le gradient boosting de 0,0053.

**Pourquoi pas un réseau de neurones ou un DQN ?**
Le jeu a 280 états : le tabulaire suffit et permet un oracle exact. En sabot fini, la référence
linéaire suffit pour répondre à la question posée (information contre séparation). Une
approximation non linéaire pourrait changer le niveau final ; c'est une limite déclarée.

**Comment savez-vous que l'avantage vient de l'information et pas d'un meilleur algorithme ?**
Deux expériences. En paquet infini, un RL qui estime tout le modèle à partir des parties et le
résout exactement fait jeu égal avec le Q-learning : exploiter mieux les parties ne suffit pas.
En sabot fini, à information égale, l'hybride est rattrapé par le Q-learning linéaire.

**Le réglage des hyperparamètres ne favorise-t-il pas l'hybride ?**
Le réglage a été fait sur des graines distinctes de toute évaluation. Refait sans la loi exacte
du croupier, il choisit la même configuration. Le standard « commun », avec la même exploration
que l'hybride, a un regret voisin du standard réglé.

**Pourquoi l'hybride perd-il avec 1 000 mains ?**
Il converge vers l'optimum de son modèle, dont le regret limite vaut 0,0030 avec 1 000 mains ;
le standard, lui, converge vers l'optimum vrai et finit par passer dessous (−0,0020 à
200 000 parties). C'est démontré (modele2.pdf) et mesuré (§5.4).

**Vos résultats en sabot fini ne sont-ils pas du sur-ajustement à vos analyses ?**
C'est le risque d'une analyse exploratoire, d'où la confirmation : plan commité avant le
lancement, données, graines et sabots neufs, aucun nouveau réglage, correction de Bonferroni.
Les quatre hypothèses sont confirmées avec des effets presque identiques.

**Que garantit la théorie en sabot fini ?**
Une borne (§6.4) : le regret de tout agent qui compare deux estimations est contrôlé par les
erreurs de ces estimations, pondérées par la fréquence des états. Elle explique l'ordre des
agents, pas leur niveau, et aucune convergence n'est garantie avec un état comprimé ou une
approximation linéaire.

**Peut-on battre le casino avec ça ?**
Non. Même la stratégie optimale perd 4,66 centimes par euro misé, et nos meilleurs agents en
sabot fini gagnent environ 0,005 par manche sur la stratégie de base, sans changer la mise.
Le comptage rapporte surtout en faisant varier la mise, ce que le projet n'étudie pas.

**Qui a fait quoi ?**
Voir la section « Auteurs » du README : Wassim Ghnaya a conçu l'environnement, le Q-learning et
le modèle 2 ; Alessandro Daouk le modèle 1, l'agent assemblé et les campagnes de comparaison.
