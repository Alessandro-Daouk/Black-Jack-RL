# Dimensionnement après pilote, avant tout test final

Le pilote de validation (`results/v2/pilot/design.json`) donne un écart-type de
0,000843598 entre quatre différences de grappes. La règle préétablie, avec inflation
1,5, impose **28 grappes** pour une demi-largeur projetée de **0,000490670** sur
le contraste principal à 20 000 mains et 100 000 épisodes. Les 28 grappes sont conservées.

Précision du périmètre secondaire avant exécution : les surfaces, ablations et le
prolongement à 200 000 épisodes sont descriptifs. Leur allocation utilise la même
règle et un objectif de demi-largeur **0,001** (la différence pratique minimale),
ce qui donne **9 grappes**. Celles-ci sont les neuf premières des 28, désignées
avant toute consultation de résultats finaux. Les 19 autres grappes complètent
uniquement logistique et standard TUNED à 20 000 mains / 100 000 épisodes.

Ce dimensionnement évite d'étendre artificiellement la précision du contraste
principal à toutes les questions exploratoires. La dispersion du pilote concerne
ce contraste seulement ; il n'existe **aucune garantie** d'atteindre ±0,001 sur
les autres tailles ou ablations. Leurs IC effectifs, y compris larges, seront publiés.
Les trajectoires à budget différent ne sont pas des répétitions indépendantes.

Le plan final comporte 9×2×19 + 19×2×2 = **418 entraînements**,
soit **76 000 000 épisodes**. L'ordre de lancement reste mélangé et les mesures
restent séquentielles, avec un thread numérique. Pas de changement des graines,
des hyperparamètres, des seuils, des conventions de coût ou du modèle livré.
La configuration exécutable est `experiments/v2/final_frozen.json`.

Le protocole initial demeure accessible dans le commit `b53d918`. Cet addendum
explicite une adaptation faite sur validation ; ce n'est pas un résultat de test.
