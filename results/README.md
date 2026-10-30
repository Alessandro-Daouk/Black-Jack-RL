# Résultats et statut scientifique

`supervised/`, `q_learning/` et `hybrid/` contiennent les expériences historiques,
conservées sans réécriture de leurs valeurs ou rapports générés. Les affirmations
de ces anciens rapports doivent être lues avec les corrections de l'audit v2,
notamment sur le KL brut, les unités de coût et la portée des intervalles.

`v2/` contient les campagnes de l'audit. Un `completion.json` indique la fin
d'une campagne ; l'absence de ce fichier signifie que ses sorties sont partielles.
Les configurations et empreintes du code lors de chaque lancement figurent dans
`provenance.json`. Les résultats finaux ne servent pas à sélectionner l'agent livré.

Les scripts refusent les dossiers non vides. Pour reproduire, utiliser
`results/reproduction/`, ignoré par Git. Les fichiers v2 sont conservés octet par
octet dans Git (règle `.gitattributes`) afin que les empreintes restent valables
sous Linux et Windows.

Lecture scientifique actuelle : [rapport](../docs/rapport_scientifique.md),
[protocole](../docs/protocole_v2.md) et [addendum avant test](../docs/protocole_v2_addendum.md).
