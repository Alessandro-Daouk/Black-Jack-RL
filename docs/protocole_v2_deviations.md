# Déviations et précisions postérieures au protocole v2

Ce fichier recense les changements apportés après la fixation du protocole
(`docs/protocole_v2.md`) et de son addendum. Aucun d'eux ne modifie les graines,
les budgets, les hyperparamètres, le critère principal ni le modèle livré.

## D1 — Test d'équivalence TOST sur les contrastes (analyse seulement)

Le protocole fixe une différence pratique minimale de 0,001 récompense par partie
et interdit de conclure à l'équivalence à partir d'un test non significatif. Il ne
précisait pas de procédure d'équivalence. L'analyse ajoute donc, pour chaque
contraste apparié, l'intervalle de Student à 90 % sur les grappes indépendantes et
la p-valeur du double test unilatéral de Schuirmann (TOST) avec la marge ±0,001.
L'équivalence n'est déclarée que si ce p est inférieur à 0,05, ce qui revient à un
intervalle à 90 % inclus dans la marge. Ces tests secondaires ne sont pas corrigés
pour la multiplicité ; ils sont descriptifs, comme les autres contrastes secondaires.

Chronologie : cette décision a été prise pendant l'exécution de la campagne finale,
avant toute analyse de ses résultats. Seules des lignes de progression du journal
(gain final de quelques entraînements isolés) avaient été affichées ; aucun
contraste, aucune moyenne et aucun intervalle n'avaient été calculés. La marge
elle-même est celle du protocole, fixée avant le pilote.

## D2 — Réplication E1 exécutée après la campagne finale

Le protocole n'imposait pas d'ordre entre la réplication et les campagnes finales,
mais interdisait tout entraînement concurrent pendant les mesures de temps. La
réplication des 240 entraînements historiques et la variante E6 ont donc été
lancées après la fin de la campagne finale, séquentiellement. Le script de
réplication a été modifié avant son exécution pour consigner un éventuel écart dans
`comparison.json` au lieu de s'interrompre sur une assertion, et pour écrire un
fichier `completion.json` comme les autres campagnes. Les tolérances (1e-9 sur les
gains exacts, 1e-7 sur les probabilités du prédicteur) sont inchangées.

## D3 — Vérification d'intégrité

`scripts/verify_audit.py` recalcule le gain exact de chaque Q-table archivée de la
campagne finale, contrôle les empreintes des agents et des tables, et vérifie que
l'analyse publiée porte sur ces tables. Ce script ne produit aucune mesure nouvelle.

## D4 — Description de la variante E6 à tous les budgets

Le protocole E6 ne fixait pas de budget d'analyse ; le script d'analyse, écrit avant
les mesures, ne rapportait que le contraste à 100 000 épisodes. Après lecture de ce
contraste, qui ne montre pas d'avantage établi, l'analyse a été étendue à une
description de tous les budgets de la grille, afin de ne pas réduire la variante à un
seul point. Cette extension est postérieure aux résultats : ses intervalles sont
descriptifs, non simultanés, et ne constituent pas un test préétabli.

## D5 — Correction d'un défaut de l'analyse avant sa version publiée

Une exécution d'essai de l'analyse, dans un dossier temporaire, a révélé que le
nombre de grappes écrasait la taille supervisée dans les tableaux de surface et de
coûts appariés (la figure échouait et le tableau des coûts du rapport généré était
vide). La correction ne change aucun calcul statistique ; l'analyse publiée a été
produite après elle. L'échelle de couleur de la surface a aussi été saturée à ±0,005
pour rendre lisibles les écarts proches de la marge pratique.
