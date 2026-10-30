# État de l'art

Recherche effectuée le 30 octobre 2026. Les sources ci-dessous sont les articles
originaux, les ouvrages des auteurs ou leurs sites institutionnels. La date d'un dépôt
arXiv ne remplace pas celle de la publication. Cette recherche ciblée n'est pas une
revue systématique exhaustive.

## Comparaison critique

| Travail | Idée pertinente | Recouvrement avec le dépôt | Différence | Apport défendable ici |
|---|---|---|---|---|
| Watkins & Dayan, 1992 [1] | Q-learning tabulaire et convergence sous hypothèses | Bootstrap par maximum, apprentissage hors politique | Une action est figée dans l'hybride ; DAG absorbant à γ=1 | Spécialisation explicite des hypothèses et contrôle de leur implémentation |
| Sutton & Barto, 2018 [2] | Bellman, évaluation/contrôle, apprentissage et planification | MDP fini, oracle DP, epsilon-greedy, Expected SARSA comme comparaison | Intégrer l'issue d'une action terminale n'est pas moyenner sur l'action suivante de la politique | Séparation claire des objets estimés et des budgets |
| Sutton, 1990, Dyna [3] | Alterner expérience réelle et planification avec un modèle appris | Combiner connaissance d'une dynamique et apprentissage de valeurs | Ici pas de transitions synthétiques supplémentaires ni de boucle Dyna ; seule STAND est calculée | Cas minimal isolant une composante de valeur modélisée |
| Shelton, UAI 2002 [4] | Dynamiques partiellement connues, intégration de connaissance du domaine | Utiliser ce qui est connu plutôt que le réestimer aveuglément | PKMDP, importance sampling et optimisation de politique, pas notre Q-learning à action terminale figée | Étude contrôlée d'une architecture plus simple, sans revendication d'invention du principe |
| Blackwell, 1947 [5] | Conditionnement et réduction de variance | Remplacer un gain terminal aléatoire par son espérance exacte | Une espérance **estimée**, passée dans un maximum adaptatif, n'est pas automatiquement une Rao–Blackwellisation de toute la cible RL | Exposer les conditions de l'analogie et mesurer les erreurs de modèle |
| van Hasselt, 2010 [6] | Biais du maximum et Double Q-learning | Le maximum de valeurs bruitées peut introduire un biais | Double Q sépare sélection/évaluation ; figer STAND est un autre mécanisme | Ablations de l'information et de l'exploration, sans prétendre éliminer tout biais |
| Gneiting & Raftery, 2007 [7] | Scores strictement propres et calibration | Log-loss et Brier pour des probabilités utiles à une décision | Les métriques de classe seules ne caractérisent pas la qualité décisionnelle | Relier risque probabiliste, erreur de STAND, marges et regret exact |
| Puterman, 1994 [8] ; Baldwin et al., 1956 [15] | MDP finis, programmation dynamique ; stratégie de base du Blackjack calculée exactement | Oracle par récurrence sur le DAG | L'oracle n'est ni nouveau ni appris ; il sert uniquement à évaluer | Second oracle indépendant et contrôles Monte-Carlo, pas une nouvelle stratégie |
| Tsitsiklis, 1994 [9] ; Jaakkola, Jordan & Singh, 1994 [10] | Convergence du Q-learning par approximation stochastique, y compris non actualisé avec politiques propres | Convergence de l'hybride vers l'optimum du modèle estimé | Leur cadre est général ; ici une induction finie sur le DAG suffit | Preuve spécialisée, hypothèses explicitées une à une |
| Singh & Yee, 1994 [11] ; Kakade & Langford, 2002 [12] | Perte d'une politique gloutonne approchée ; lemme de différence de performance | Bornes de regret 2η et identité d'occupation | Bornes classiques en 1/(1−γ), inapplicables telles quelles à γ=1 | Borne 2η exploitant une seule récompense terminale perturbée ; borne locale pondérée par occupation |
| van Seijen et al., 2009 [13] ; Sutton & Barto §6.8 [2] | Expected Sarsa ; valeurs d'après-états | Remplacer un échantillon par une espérance dans la cible | Expected Sarsa moyenne sur la politique ; ici l'espérance porte sur le croupier, estimée hors ligne | Mesure de ce que l'espérance estimée apporte et coûte |
| Ng, Harada & Russell, 1999 [14] | Façonnage de récompense par potentiel, invariance de la politique optimale | Injection d'une connaissance préalable dans l'apprentissage | Un potentiel préserve la politique optimale ; une STAND erronée peut la changer | Ablation par perturbation contrôlée de STAND et regret limite exact |
| Schuirmann, 1987 [16] | Double test unilatéral (TOST) d'équivalence | Conclure à l'absence de différence pratique | Discipline statistique, pas méthode RL | Marge fixée avant les mesures et TOST sur grappes indépendantes |

## Portée de cette recherche

Les bornes de regret du dépôt (2η et borne locale pondérée par l'occupation) appliquent
des arguments usuels de perturbation et de différence de performance ; elles ne sont pas
présentées comme des théorèmes nouveaux. Shelton (2002) est un antécédent conceptuel
proche, sans que cette recherche ciblée prouve qu'il soit le plus proche de toute la
littérature, ni qu'aucun travail antérieur ne fixe exactement la valeur d'une action
terminale. Aucune priorité scientifique n'est revendiquée. La contribution du projet est
discutée dans le [rapport](rapport_scientifique.md#6-discussion).

## Références vérifiées et notes de version

1. **C. J. C. H. Watkins & P. Dayan (1992).** *Q-learning*, Machine Learning 8,
   279–292. [Article sur le site de Peter Dayan](https://www.gatsby.ucl.ac.uk/~dayan/papers/cjch.pdf).
   L'article traite la convergence tabulaire et discute les environnements absorbants
   non actualisés ; notre preuve spécifique au DAG explicite les hypothèses utilisées.
2. **R. S. Sutton & A. G. Barto (2018).** *Reinforcement Learning: An Introduction*,
   2e édition, MIT Press, chapitres 3–6 et 8.
   [Éditeur, édition publiée le 13 novembre 2018](https://mitpress.mit.edu/9780262039246/reinforcement-learning/).
   [Manuscrit des auteurs daté du 21 mars 2018](https://www.incompleteideas.net/book/bookdraft2018mar21.pdf).
   Un miroir Stanford consulté est explicitement une **version en cours 2014–2015**,
   pas l'édition finale ; les numéros de page de ces versions ne sont pas mélangés.
3. **R. S. Sutton (1990).** *Integrated Architectures for Learning, Planning, and
   Reacting Based on Approximating Dynamic Programming*, ICML, 216–224.
   [Article original](https://www.derongliu.org/adp/adp-cdrom/refs/sutton19900216.pdf).
   Cette référence est l'article ICML 1990 ; ne pas le confondre avec l'article
   SIGART 1991 au titre voisin.
4. **C. R. Shelton (2002).** *Reinforcement Learning with Partially Known World
   Dynamics*, UAI, 461–468.
   [Texte et notice](https://arxiv.org/abs/1301.0601), [article](https://arxiv.org/pdf/1301.0601).
   Le PDF porte UAI 2002 ; sa mise en ligne arXiv ultérieure ne date pas la contribution.
5. **D. Blackwell (1947).** *Conditional Expectation and Unbiased Sequential
   Estimation*, Annals of Mathematical Statistics 18(1), 105–110.
   [DOI](https://doi.org/10.1214/aoms/1177730497),
   [texte original, miroir institutionnel](https://ds.dfci.harvard.edu/~gp/talks/Blackwell12.pdf).
6. **H. van Hasselt (2010).** *Double Q-learning*, Advances in Neural Information
   Processing Systems 23.
   [Actes officiels et article](https://proceedings.neurips.cc/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html).
7. **T. Gneiting & A. E. Raftery (2007).** *Strictly Proper Scoring Rules,
   Prediction, and Estimation*, JASA 102(477), 359–378.
   [DOI](https://doi.org/10.1198/016214506000001437),
   [article sur le site de l'auteur](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf).

8. **M. L. Puterman (1994).** *Markov Decision Processes: Discrete Stochastic
   Dynamic Programming*, Wiley. [DOI](https://doi.org/10.1002/9780470316887).
9. **J. N. Tsitsiklis (1994).** *Asynchronous Stochastic Approximation and
   Q-Learning*, Machine Learning 16, 185–202.
   [DOI](https://doi.org/10.1007/BF00993306),
   [article sur le site de l'auteur](https://www.mit.edu/people/jnt/Papers/J052-94-jnt-q.pdf).
10. **T. Jaakkola, M. I. Jordan & S. P. Singh (1994).** *On the Convergence of
    Stochastic Iterative Dynamic Programming Algorithms*, Neural Computation 6(6),
    1185–1201. [Notice](https://mlanthology.org/neco/1994/jaakkola1994neco-convergence/).
    Une version courte au titre différent a paru à NIPS 1993 ; elle n'est pas citée ici.
11. **S. P. Singh & R. C. Yee (1994).** *An Upper Bound on the Loss from
    Approximate Optimal-Value Functions*, Machine Learning 16, 227–233.
    [Notice](https://mlanthology.org/mlj/1994/singh1994mlj-upper/). Deux DOI
    différents circulent pour cet article ; aucun n'est reproduit faute de vérification.
12. **S. Kakade & J. Langford (2002).** *Approximately Optimal Approximate
    Reinforcement Learning*, ICML, 267–274.
    [Notice](https://mlanthology.org/icml/2002/kakade2002icml-approximately/).
    Le lemme y est énoncé dans le cas actualisé ; le dépôt utilise sa version épisodique.
13. **H. van Seijen, H. van Hasselt, S. Whiteson & M. Wiering (2009).** *A
    Theoretical and Empirical Analysis of Expected Sarsa*, IEEE ADPRL, 177–184.
    [DOI](https://doi.org/10.1109/ADPRL.2009.4927542).
14. **A. Y. Ng, D. Harada & S. Russell (1999).** *Policy Invariance under Reward
    Transformations: Theory and Application to Reward Shaping*, ICML, 278–287.
    [Notice](https://mlanthology.org/icml/1999/ng1999icml-policy/).
15. **R. R. Baldwin, W. E. Cantey, H. Maisel & J. P. McDermott (1956).** *The
    Optimum Strategy in Blackjack*, JASA 51(275), 429–439. Règles de casino
    différentes de Blackjack-v1 ; citée pour l'antériorité du calcul exact.
16. **D. J. Schuirmann (1987).** *A Comparison of the Two One-Sided Tests
    Procedure and the Power Approach for Assessing the Equivalence of Average
    Bioavailability*, Journal of Pharmacokinetics and Biopharmaceutics 15(6),
    657–680. [DOI](https://doi.org/10.1007/BF01068419).

Le résultat de risque lissé en (K−1)/(2m) est démontré dans `modele1.tex` ; nous
n'avons pas pu vérifier dans une source primaire l'énoncé exact attribué parfois à
la littérature sur les estimateurs « add-β », et ne citons donc aucune source pour lui.
