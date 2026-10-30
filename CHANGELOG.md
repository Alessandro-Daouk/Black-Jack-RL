# Journal des modifications

Les versions sont désignées par leur commit de référence. La version historique
reste lisible dans Git ; ses modèles, résultats et protocoles ne sont pas réécrits.

## Audit scientifique v2 — branche `audit/scientifique-v2` (30 octobre 2026)

Référence auditée : `9c3d6318fe3d499e4f8e50b5680df1aabf459ff8` (`main`).
L'agent livré, le prédicteur livré, les sources de Wassim Ghnaya et les résultats
historiques sont inchangés ; `scripts.verify_release` le contrôle.

### Démarche et protocole

- Audit initial écrit avant toute correction ni nouvelle campagne
  (`docs/audit_initial.md`), avec l'environnement et les empreintes de départ.
- Protocole v2 daté (`docs/protocole_v2.md`), pilote sur graines de validation,
  dimensionnement à 28 grappes supervisées indépendantes décidé avant le test
  (`docs/protocole_v2_addendum.md`), configuration exécutable figée
  (`experiments/v2/final_frozen.json`). Déviations ultérieures consignées
  séparément (`docs/protocole_v2_deviations.md`).

### Corrections mathématiques

- `modele1.tex` : l'espérance du KL des fréquences brutes est infinie à toute
  taille finie ; la loi en 25/n est démontrée pour les estimateurs lissés de
  Dirichlet (Jeffreys, Laplace) et le développement des fréquences brutes est
  énoncé comme développement aléatoire. Objectif exact de scikit-learn 1.9.1
  vérifié dans le source installé ; convexité énoncée sur le quotient par la
  direction d'invariance du biais ; relations stationnaires et modèle saturé.
- `modele2.tex` : horizon corrigé (20 décisions au plus, et non 18 : une main de
  deux as a un total dur de 2) ; convergence énoncée vers l'optimum du modèle
  estimé, avec ses hypothèses de visitation, de pas et de bruit ; borne de regret
  limite ramenée de 36η à 2η, borne locale pondérée par occupation, identité de
  différence de performance et borne pour un agent entraîné en budget fini ;
  mécanismes de réduction de variance et de biais qualifiés comme plausibles et
  non garantis.
- Bibliographie vérifiée sur sources primaires et intégrée aux deux documents ;
  positionnement critique dans `docs/etat_de_l_art.md`.

### Code et vérification

- `src/audit/` : oracle indépendant (propagation de masse, mains explicites,
  système triangulaire), instrumentation des coûts sans modifier les trajectoires
  du Q-learning historique, pilote, campagne finale, diagnostics, réplication,
  variante à cartes uniformes et analyse statistique par grappes.
- `src/supervised/evaluate.py` : le KL exact n'est plus écrêté ; la version écrêtée
  est rapportée sous un autre nom. Les résultats historiques sans zéro sont inchangés.
- Les scripts de campagne refusent d'écrire dans un dossier non vide.
- `scripts/verify_audit.py` : recalcul des gains exacts des 418 agents archivés,
  empreintes des tables, lien entre analyse et mesures.
- `scripts/build_docs.py` : compilation reproductible des deux PDF, refus des
  avertissements de références et de débordements.
- Tests ajoutés : oracle contre Gymnasium sur toutes les transitions atteignables,
  masses, comptage physique des cartes, variante uniforme, bornes et identité
  d'occupation, censure des temps d'atteinte, sélection des coûts.
- CI : vérification de l'audit et compilation LaTeX en mode contrôle.
- `.gitattributes` : les fichiers de `results/v2/` sont conservés octet par octet
  pour que leurs empreintes restent valables sous Windows.

### Campagnes et résultats v2

- Réplication stricte des 240 entraînements historiques et du modèle 1 : écart nul.
- Campagne finale : 28 jeux supervisés indépendants, 418 entraînements, 76 millions
  d'épisodes. Critère principal : +0,00214 [+0,00188 ; +0,00240] pour l'hybride.
- Régions de perte de l'hybride identifiées (1 000 mains et budgets RL longs ;
  acquisition qui consomme le budget) ; coûts mesurés en secondes et en cartes.
- Équivalence logistique / fréquences établie par TOST à ±0,001.
- Variante à cartes uniformes : avantage aux petits budgets seulement.

### Défauts corrigés pendant l'audit

- `modele2.tex` ne compilait pas (macro `\TV` non définie) ; PDF régénérés.
- Analyse : la taille supervisée était écrasée par le nombre de grappes dans les
  tableaux de surface et de coûts ; corrigé avant l'analyse publiée (déviation D5).
- Réplication : une assertion aurait interrompu la campagne au lieu de consigner un
  écart ; remplacée par un enregistrement explicite (déviation D2).

### Documentation

- README restructuré autour de la question de recherche, des résultats v2 avec
  incertitude, des coûts et des limites ; rapport scientifique, documentation de
  reproductibilité et réponse point par point au cahier des charges ajoutés.
- `CONTRIBUTING.md` ajouté. Aucune licence n'est ajoutée : ce choix revient aux auteurs.

## Versions antérieures

- `9c3d631` : fondements mathématiques du modèle 1.
- `ef79edc` : documentation réunie dans un README unique.
- `2003c5a` : intégration de l'hybride supervisé, campagnes reproductibles.
- `58f5ce3`, `0780a4f` : modèle 2 de Wassim Ghnaya (Q-learning standard et hybride,
  référence exacte, réglage, ablation).
