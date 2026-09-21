# Contrôle de remplissage corrigé — 18 septembre 2026

**Le contrôle laisse encore passer trop peu de cas : la couverture photographique
atteint 62,5 % pour un objectif de 80 %. Aucune erreur forte n'est acceptée.**

## Le défaut trouvé dans le contrôle précédent

Le contrôle de la veille masquait le trou d'origine **en plus** de la zone témoin.
L'essai était donc plus difficile que la tâche réelle. Sur le lot v3, dix des dix-huit
refus portaient sur des cas dont l'erreur réelle était inférieure à 12/255 : le
contrôle refusait du contenu parfaitement reconstructible.

Deux corrections :

- un seul bloc est masqué pour l'essai ; le contenu du trou vient du remplissage
  candidat, jamais des vrais pixels cachés ;
- les zones témoins sont choisies par similarité de contexte. L'anneau connu autour
  du trou est comparé au même anneau décalé, et seuls les sites dont l'écart reste
  sous 18/255 sont utilisés.

La recherche est bornée : 120 sites candidats au plus, les plus proches d'abord,
et quatre contrôles au maximum. Aucun pixel masqué n'est lu pour choisir un site.

## Calibration synthétique à vérité connue

`scripts/calibrate_inpaint_guard.py` génère six familles déterministes, sans
licence ni téléchargement. Une étiquette indépendante est calculée par une
interpolation simple : un cas est « reconstructible » si le meilleur des deux
baselines déterministes reste sous 12/255.

| Famille | Baseline simple | Moteur par défaut |
|---|---:|---:|
| Dégradé | 0,52 | 6/6 acceptés, erreur 0,83 |
| Rampe | 0,94 | 6/6 acceptés, erreur 1,74 |
| Rayures, période 6 | 0,00 | 6/6 acceptés, erreur 4,73 |
| Damier, blocs de 8 | 7,89 | 0/6 acceptés |
| Bruit flouté | 18,21 | 0/6 acceptés |
| Bruit blanc | 60,95 | 0/6 acceptés |

Le moteur par défaut passe les trois critères : couverture des cas
reconstructibles **81,8 %** (18/22), acceptation du bruit **0 %** (0/6), et aucune
erreur acceptée au-dessus de 12/255. Le damier est refusé alors que l'interpolation
simple y arrive : le remplissage par patches échoue réellement sur un bloc de 7
pixels dans un motif de période 8, ce que l'essai détecte correctement.

## Lot photographique inédit v3

Quatre photographies nouvelles, licences et SHA-256 documentés dans
`benchmarks/manifests/visual-quality-v3.json` : appareil photo (CC0, Lav Varshney),
pièces de monnaie (Brooklyn Museum, sans restriction connue), champ profond Hubble
(NASA, domaine public) et image de texte (domaine public). Trois positions et deux
tailles de trou donnent 24 cas.

| Configuration | Acceptés | Couverture | Erreurs fortes acceptées | Erreur moyenne | Gain / voisin | p95 |
|---|---:|---:|---:|---:|---:|---:|
| Moteur par défaut, 4 contrôles | 12/24 | 50,0 % | **0** | 3,11 | 22,5 % | 0,88 s |
| Sélecteur, 4 contrôles, accord requis | 12/24 | 50,0 % | **0** | 2,74 | 31,8 % | 0,62 s |
| **Sélecteur, 2 contrôles, accord requis** | **15/24** | **62,5 %** | **0** | 4,23 | 25,5 % | 0,74 s |
| Sélecteur, 2 contrôles, un seul survivant | 16/24 | 66,7 % | 1 | 7,79 | 13,6 % | 0,30 s |

Toutes les configurations échouent le critère de couverture. Les trois premières
passent le critère d'erreur forte ; la dernière échoue à cause d'un cas précis :
un trou de 3 × 3 pixels au centre du champ profond Hubble, qui masque une étoile
invisible depuis le bord. Aucun essai sur les pixels visibles ne peut détecter cet
objet. C'est la raison pour laquelle le sélecteur exige désormais que **deux
méthodes indépendantes** valident chacune leur remplissage : la copie de patches
échoue sur ce cas et bloque la proposition.

Sur les 24 cas, quatre refus sont pleinement justifiés (erreur réelle > 25/255),
quatre autres sont des refus modérés, et le sélecteur ne commet plus qu'un seul
refus abusif contre dix pour le contrôle précédent.

## Décision

- Le moteur par défaut conserve quatre contrôles et reste consultatif.
- Le sélecteur expérimental `bellium/hybrid/adaptive-inpaint:experimental-v1`
  choisit entre copie de patches et interpolation entre bords connus. Il n'est pas
  inscrit au registre, ne reçoit aucune autorité et n'est pas utilisé par l'adaptateur
  Pillow.
- Aucun seuil n'a été relevé pour faire passer un critère. Les poids sont inchangés.
- La couverture photographique reste la limite documentée : **62,5 %** pour un
  objectif de 80 %.

Le lot v3 a servi au diagnostic de ce correctif : il devient un lot de
développement. Toute nouvelle modification devra être validée sur un lot inédit,
par exemple les images `eagle`, `skin` et `lily` disponibles dans le dépôt de
données scikit-image avec leurs licences.

## Preuves

- `evaluation-2026-09-18/synthetic-final-patch.json` : calibration du
  moteur par défaut, trois critères passés.
- `evaluation-2026-09-18/visual-v3-final-adaptive` : sélecteur retenu.
- `visual-v3-patch-r7`, `visual-v3-adaptive-r8`, `visual-v3-adaptive-r5` :
  configurations comparées, conservées pour la traçabilité du compromis.
- `visual-v3-patch` : exécution avec le contrôle défectueux, qui documente les
  dix refus abusifs.

```sh
python scripts/calibrate_inpaint_guard.py --output <nouveau-fichier> --engine patch
python scripts/evaluate_visual_quality.py --manifest benchmarks/manifests/visual-quality-v3.json --cache <cache> --output <nouveau-dossier> --engine adaptive
```
