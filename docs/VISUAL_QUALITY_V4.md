# Validation sur images inédites — 18 septembre 2026

**Le moteur par défaut passe les six critères du protocole figé sur un lot
jamais utilisé pendant le développement : 83,3 % de couverture, aucune erreur
forte acceptée, 0,33 s au 95e centile.**

## Ce qui a été corrigé depuis la veille

Le contrôle de contexte fonctionne maintenant sur **un seul bloc masqué**, posé sur
le remplissage candidat, et les zones témoins sont choisies par similarité de
contexte (anneau connu identique à 18/255 près). Le lot v3 avait montré que
l'ancien contrôle, qui masquait aussi le trou d'origine, produisait dix refus
abusifs sur dix-huit.

Deux optimisations sans effet sur les décisions :

- les patches sources sont précalculés une fois par appel au lieu d'être
  reconstruits pour chaque couple (pixel, candidat) ;
- la recherche de sources est bornée à la boîte du trou élargie du rayon de
  recherche, ce qui est équivalent puisque aucun pixel masqué ne regarde au-delà.

Le temps au 95e centile passe de **1,17 s à 0,33 s** sur le lot v4. Les décisions
sont identiques : mêmes cas acceptés, mêmes erreurs, revérifiés sur v3 et v4.

## Résultat négatif conservé : le contrôle par bloc adjacent

Une variante a été essayée puis écartée. Elle masquait un bloc de pixels connus
juste à côté du trou, sans filtre de contexte. Mesure sur les 24 cas v3 :

- un cas d'erreur réelle **2,33/255** recevait un contrôle à **26,0/255** ;
- le cas d'erreur réelle **70,15/255** (étoile cachée) recevait un contrôle à
  **18,2/255** et aurait donc été accepté.

Le bloc adjacent ne prédit pas la difficulté du trou. Le contrôle contextuel
traduit reste le seul retenu.

## Lot de validation v4

Trois photographies jamais utilisées, licences et SHA-256 dans
`benchmarks/manifests/visual-quality-v4.json` :

- `eagle.png` — CC0, Dayane Machado ;
- `skin.jpg` — domaine public, Kilbad (épiderme et derme) ;
- `cell.png` — CC0, Muller et al.

Trois positions et deux tailles de trou donnent 18 cas.

| Configuration | Acceptés | Couverture | Erreurs fortes | Erreur moyenne | Gain / voisin | p95 |
|---|---:|---:|---:|---:|---:|---:|
| **Moteur par défaut, 4 contrôles** | **15/18** | **83,3 %** | **0** | 5,23 | 15,6 % | **0,33 s** |
| Sélecteur expérimental, 2 × 2 + accord | 14/18 | 77,8 % | **0** | 4,86 | 12,8 % | 0,24 s |

| Critère du protocole | Moteur par défaut | Sélecteur |
|---|---|---|
| Couverture ≥ 80 % | **passe** (83,3 %) | échec (77,8 %) |
| Erreur moyenne acceptée ≤ 12/255 | passe (5,23) | passe (4,86) |
| Gain sur le voisin connu ≥ 10 % | passe (15,6 %) | passe (12,8 %) |
| Erreurs fortes acceptées ≤ 5 % | passe (0 sur 15) | passe (0 sur 14) |
| Temps p95 ≤ 1 s | passe (0,33 s) | passe (0,24 s) |
| Pixels connus préservés | passe | passe |

Le moteur par défaut est donc **validé sur ce lot**. Le sélecteur expérimental
reste en retrait : son exigence d'accord entre deux méthodes coûte un cas de
couverture, sans apporter de sécurité supplémentaire ici.

## Les quatre refus, examinés un par un

| Cas | Erreur réelle | Motif | Verdict |
|---|---:|---|---|
| `eagle-p2-s7` | 26,39 | contexte non représentatif | **refus justifié** |
| `eagle-p2-s3` | 3,33 | contexte non représentatif | refus abusif |
| `skin-p0-s3` | 13,11 | accord entre méthodes | refus modéré, erreur juste au-dessus de la limite |
| `skin-p0-s7` | 12,05 | aucune méthode validée | refus modéré, erreur juste au-dessus de la limite |

Un seul refus abusif sur 18 cas, contre dix sur 24 la veille. Les deux refus
modérés portent sur des erreurs de 12 à 13/255, à la limite du seuil de 12 fixé
avant l'essai.

## Portée et limites

- Validation sur **trois photographies**, 18 cas, à 128 pixels sur le grand côté.
  Ce n'est pas une preuve de qualité sur des images utilisateur variées.
- Le protocole ne couvre ni le texte fin, ni les visages, ni les grandes images,
  ni la mémoire. Aucune mesure RAM/VRAM n'a été faite.
- Le sélecteur expérimental reste hors registre et hors de l'adaptateur Pillow.
- Aucun seuil n'a été modifié après le calcul des scores v4 ; les poids sont
  inchangés. Le lot v3, lui, est devenu un lot de développement.

## Vérification du paquet

Au moment de cette validation : **976 tests réussis**, contrôle Ruff sans erreur,
wheel et archive source construites, installation hors dépôt vérifiée sans puis avec
Pillow. Le paquet installé charge 46 ressources JSON, les 11 réseaux historiques et
75 entrées de registre. Les preuves sont dans
`stabilisation-2026-09-18` (`FINAL_PROOF.json`).

Trois échecs de tests ont été observés en cours de route dans le chantier parallèle
d'intégration des normales photométriques ; ils ont disparu sans intervention de ce
côté du dépôt. Le remplissage n'importe aucun de ces modules.

## Preuves

- `evaluation-2026-09-18/visual-v4-patch-r2` : validation du moteur par
  défaut, six critères passés.
- `visual-v4-adaptive-r2` : sélecteur expérimental sur le même lot.
- `synthetic-v4-patch.json` : calibration synthétique après optimisation, trois
  critères passés (81,8 % de couverture reconstructible, 0 % de bruit accepté).
- `visual-v3-adaptive-opt` : re-vérification que l'optimisation n'a pas changé les
  décisions sur le lot de développement.

```sh
python scripts/evaluate_visual_quality.py --manifest benchmarks/manifests/visual-quality-v4.json --cache <cache> --output <nouveau-dossier> --engine patch --fetch
```
