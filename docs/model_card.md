# Fiche du modèle livré

## Identité et utilisation

- Projet : Blackjack supervisé + Q-learning, Alessandro et Wassim.
- Usage : démonstration et étude pédagogique dans `Blackjack-v1` aux règles fixées.
- Prédicteur : `models/supervised/dealer_model.joblib`.
- Agent : `models/hybrid/agent_tuned.npz`.
- Empreintes : `models/manifest.json`, vérifiables avec `python -m scripts.verify_release`.
- Chargement : `src.hybrid_agent.load_agent()`, puis `agent.decide((total, carte, as))`.

Entrée valide : total joueur 4–21, carte visible 1–10, as utilisable 0/1 ;
avec as utilisable, le total doit être au moins 12. Sortie : action HIT/STAND,
valeur estimée de chaque action et nombre de visites HIT à l'entraînement.
Les six probabilités du croupier sont également consultables. Aucune carte cachée en entrée.

## Apprentissage et choix

Régression logistique multinomiale, one-hot, C=0.1 ; train 60 000 mains/graine 42,
validation 12 000/43, test 12 000/123. C choisi sur validation, sans refit.
Prédicteur ensuite figé ; Q_HIT appris sur 100 000 épisodes, graine 0.
RL TUNED : eps0=eps_min=1 pendant l'apprentissage, gamma=1, omega=1,
évaluation et jeu sans exploration. Sources RL de Wassim : `58f5ce3`.
TUNED a été retenu sur les graines de validation 200–219 ; graine finale 0 fixée.

## Mesures

- Prédicteur : log-loss test 1.651587, Brier multiclasse 0.778452, accuracy 36.49 %.
- Erreur maximale d'une probabilité : 0.019042 ; excès exact de log-loss : 0.000429.
- Agent livré : J exact −0.047206653, optimum −0.046555977, écart 0.000650676.
- Test de 10 000 parties : gain −0.0545, IC approximatif à 95 % [−0.0731, −0.0359].

Les nouvelles expériences ne montrent pas d'avantage net du classificateur sur
les fréquences empiriques. L'hybride dépasse en moyenne le standard à budget RL
égal ; l'avantage dépend du coût imputé aux données supervisées. Voir le [rapport](rapport.md).

## Limites et entretien

Paquet infini, arrêt sur 17 souple, natural=False, sab=False, actions HIT/STAND.
Aucune validation sur d'autres règles. Le gain espéré reste négatif, même pour
la politique optimale. Les résultats ne promettent pas de gagner une partie.
L'étude de taille des données porte sur cinq jeux supervisés, et les comparaisons
RL sur un prédicteur figé ; sa variabilité n'est pas propagée dans le RL.

Changer le prédicteur nécessite une nouvelle Q-table cohérente ; le chargeur refuse
un fichier supervisé dont l'empreinte diffère de celle attendue. Changer les règles
nécessite de revoir les données, les valeurs et l'évaluation. Changer seulement
l'interface n'impose pas de réentraînement. Charger uniquement les fichiers joblib
de confiance produits ou livrés par ce projet.
