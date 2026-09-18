# Stabilisation après audit — 15 septembre 2026

Les corrections réunissent les API publiées au commit
`9ab60d91a9b78829526c3a99f35d87fcadf56bc2` et les spécialistes locaux. Les
fichiers locaux existants ont été conservés lors de l'import ; les changements
concurrents de prosodie, recadrage, bulles et autres spécialistes restent présents.
Voir `COMPATIBILITY_IMPORT.json` pour les chemins ajoutés et leur origine.

## Corrections de l'audit

| Constat | Correction locale |
|---|---|
| A01 : développements séparés | Modules publiés réunis dans la même arborescence ; API de compatibilité conservées, inpainting Pillow délégué au cœur corrigé. L'historique Git n'est pas encore publié/réconcilié. |
| A02 : paquet non constructible | Découverte explicite des paquets, backend de construction, JSON embarqués dans la wheel, extra Pillow et vérification d'installation hors dépôt. |
| A03 : remplissage incomplet | Propagation par fronts successifs, distance sur pixels connus uniquement, masque résiduel et abstention si un trou reste. |
| A04 : alpha perdu | Transparence RGBA préservée ; les pixels invisibles ne peuvent pas devenir opaques. |
| A05 : fausse identité de prononciation | Égalité explicite des séquences pour certifier ; insertions, suppressions et substitutions exposées. Le DTW reste une mesure de similarité. |
| A06 : langue réécrite | Inventaire incohérent rejeté. La langue d'origine reste immuable. |
| A07 : veto dépendant du modèle | Retour immédiat du veto avant features inutiles et chargement de modèle. |
| A08 : phonèmes ex aequo certifiés | Ambiguïté signalée avec abstention ; certification limitée à un match exact unique avec support attesté. |
| A09 : nombres/modes invalides | Contrat strict, NaN/Inf rejetés dans les features et modèles, dimensions/activations validées. |
| A10 : tests dispersés | Scénarios historiques exécutés par pytest ; commandes documentées corrigées ; workflow Windows/Linux ajouté. |
| A11 : détourage lent | Couleurs de bord regroupées en conservant les multiplicités exactes ; sélection bornée, cache local et plafond de couleurs. |
| A12 : extracteurs divergents | Reconnaissance des expressions JavaScript et traces de pile restaurée, tests de parité. |
| A13 : calibration inopérante | Température consommée par les classifieurs nommés, fichiers liés aux octets du modèle ; entrée de calibration automatique explicitement désactivée, sans import absent. |
| A14 : paramètres ignorés | Adoucissement spatial et marge appliqués ; normalisation refusée avec métriques si extraction incertaine ; descriptions rectifiées. |
| A15 : mémoire fragile | Lecture et écriture bornées, validation des records et scores ; lignes invalides ignorées, registre plein refusé. Un seul écrivain est supposé. |
| A16 : faux support par duplication | Exemplaires d'outils dédupliqués par ID ; contradictions sur un ID rejetées. |
| A17 : décisions ambiguës | Seuil de mutation harmonisé, criticité élevée traitée par règle ; abstention du second avis propagée. |
| A18 : labels visuels contradictoires | Contradictions vérifiées avant la nouveauté, avec abstention dans les deux cas. |
| A19 : mauvais modèle écrasant le bon | Seuil et score fini vérifiés avant sauvegarde, remplacement atomique ; tests sans entraînement lourd. |

Les modules publiés depuis l'audit ont aussi été examinés lors de leur intégration :
le routeur de capacités s'abstient s'il ne trouve aucun outil pertinent et ne
propose pas de repli exclu par les contraintes ; l'inpainting Pillow ne prétend
plus remplir une image entièrement masquée ; la mémoire phonémique de compatibilité
demande une langue pour inférer dans une mémoire multilingue.

## Provenance et licences

Les poids importés n'ont pas été réentraînés ni remplacés. Leurs empreintes sont
vérifiées au chargement natif. Les fichiers d'attribution MIT et le texte complet
Apache sont présents. Les identifiants du manifeste historique sont alignés sur
l'inventaire, en conservant les anciens IDs comme métadonnées. Les chemins source
du manifeste local sont désormais relatifs au dépôt, avec une empreinte distincte
du blob Git pour documenter les conversions historiques de fins de ligne.

Le nouvel importeur exige un SHA de commit complet, lit les blobs Git épinglés,
prépare tous les fichiers avant création et refuse d'écraser une version existante.
Il ne reprend pas silencieusement les poids modifiés dans un répertoire de travail.
Les adaptations du code historique restent identifiées par ce document et les
notices ; les droits MIT des composants importés sont préservés.

## Preuves et limites

Les contre-exemples de l'audit ont d'abord échoué dans la nouvelle suite de
non-régression, puis réussi après correction. Les résultats détaillés de la
validation locale et du paquet sont conservés dans le dossier
`C:\BelliumAI\stabilisation-2026-09-15`.

Validation finale du candidat intégré : **137 tests réussis**, contrôle Ruff
réussi, wheel et archive source construites. Le paquet a été installé hors du
dépôt dans un environnement vierge sans Pillow, puis vérifié avec Pillow dans
un second environnement. Les **27 ressources JSON natives**, les **11 réseaux
historiques** et les **35 entrées du registre** ont été contrôlés ; aucune entrée
n'est autorisée à décider seule. La wheel contient aussi les 11 poids historiques
de compatibilité et leur manifeste, soit 39 fichiers JSON au total.

La reconstruction de la wheel depuis l'archive source réussit également ; les
fichiers Python et JSON sont identiques à ceux de la première wheel. Les poids
natifs et historiques présents avant les corrections ont conservé leurs octets.
Les vérifications ont été exécutées sous Windows et Python 3.14 ; la matrice CI
Windows/Linux et Python 3.10/3.14 reste à exécuter à distance.

Le benchmark reproductible mesure, sur le carré synthétique de 128 × 128,
une médiane de **0,025 s sur cinq passages**, contre **10,42 s observées pendant
l'audit**. Le calcul exact des cinq voisins est comparé à la version non compactée
sur des palettes répétées. Ce gain ne constitue pas une évaluation sur des
photographies variées.

Les deux réseaux natifs conservent 98,4 % et 99,0 % d'accord sur le nouveau jeu
synthétique de 1 000 exemples chacun. Ils imitent des règles qui obtiennent par
construction 100 % sur ces mêmes étiquettes. Aucun avantage sur données réelles,
calibrage terrain, mesure de RAM/VRAM ni intégration active dans un consommateur
n'est affirmé. La collecte automatique de logs de calibration reste indisponible.

La CI est préparée ; une réussite distante Windows/Linux ne peut être annoncée
avant sa publication et son exécution. Aucun spécialiste n'est promu en mode actif.
Les tests ne remplacent pas une validation indépendante de qualité.

## Suite du 16 septembre

Le rapport photographique v1 a été finalisé après le rétablissement de l'outil
local. Un contrôle de reconstruction du contexte est maintenant appliqué par
défaut au remplissage ; il s'abstient lorsqu'il ne peut pas vérifier sa compétence
locale. Voir [le bilan v2](VISUAL_QUALITY_V2.md) pour la comparaison indépendante
et la limite de couverture. Cette étape ne constitue pas une validation de production.

Le candidat intégré suivant passe 236 tests, Ruff, la construction de la wheel
et de l'archive source, ainsi que l'installation hors dépôt avec et sans Pillow.
Les preuves du 16 septembre sont conservées séparément de celles du 15.
