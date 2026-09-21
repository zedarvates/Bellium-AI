# Contrôle d'incertitude du remplissage — 16 septembre 2026

**Le contrôle réduit les propositions dégradées, mais l'objectif de couverture
n'est pas atteint. Le spécialiste reste expérimental et consultatif.**

## Ce qui change

Après avoir produit un candidat, le remplisseur masque temporairement jusqu'à
quatre régions visibles voisines, de la même forme que le trou. Il les reconstruit
avec le même algorithme, puis compare ces essais aux pixels réellement connus.
Il exige au moins deux essais complets, tous avec une erreur moyenne ≤12/255.
Le budget est limité à 256 pixels manquants ; une vérification impossible ou
dégradée conduit à l'abstention.

Le contrôle ne lit jamais les vrais pixels cachés dans le trou et n'entraîne
aucun modèle. Il mesure la prévisibilité du contexte, sans garantir le contenu
inconnu. La reconstruction du candidat reste celle du moteur précédent ; c'est
la sélection des résultats proposés qui change.

Le champ `confidence` vaut désormais `None` sur les candidats vérifiés ainsi.
L'ancien score de correspondance reste dans `output.support_score`, accompagné
des erreurs d'essai dans `output.quality`. Ces valeurs ne sont pas présentées
comme des probabilités calibrées. L'adaptateur Pillow conserve l'image d'entrée
et indique zéro pixel appliqué lorsqu'une proposition est refusée.

## Un nouvel essai fixé avant les résultats

Le lot v1 (café, chat, briques, herbe) sert désormais au diagnostic. Le lot v2
utilise **quatre nouvelles photographies** : astronaute, fusée, horloge et gravier.
Les sources et licences sont documentées par
[scikit-image](https://scikit-image.org/docs/0.25.x/api/skimage.data.html), avec
révision et SHA-256 dans `benchmarks/manifests/visual-quality-v2.json`.

Les mêmes trois positions, deux tailles de trou et critères de qualité du v1
produisent 24 cas. Le seuil 12 du contrôle vient du critère d'erreur précédemment
fixé ; il n'a pas été ajusté sur les nouveaux scores. Une seule variante et une
seule exécution de qualité v2 ont été évaluées. Les poids sont inchangés.

L'ancien moteur a été exécuté sur **ces mêmes cas v2**, depuis une copie dont
l'empreinte est verrouillée dans le protocole. Le rapport conserve ses résultats.

## Comparaison mesurée

| Mesure | Ancien moteur, mêmes cas v2 | Avec contrôle local |
|---|---:|---:|
| Résultats proposés | 24/24 | **11/24** |
| Cas proposés dont l'erreur dépasse 25/255 | **7/24** | **0/11** |
| Cas refusés | 0 | **13** |
| Erreur moyenne sur les cas proposés par chaque version | 17,63/255 | 1,60/255 |
| Temps p95 | 0,066 s | 0,365 s |
| Pixels connus conservés | tous | tous |

Les moyennes d'erreur ne portent pas sur les mêmes sous-ensembles : la diminution
résulte du rejet des cas risqués, et ne prouve pas une amélioration de leurs pixels.
Parmi les 13 refus figurent les sept erreurs fortes, quatre erreurs intermédiaires
et deux résultats qui étaient sous 12/255. Le contrôle est donc conservateur.

Sur les 11 cas retenus, le voisin connu le plus proche obtient 1,99/255 contre
1,60/255 pour le patch, soit 19,75 % de gain. Cette comparaison porte bien sur
les mêmes cas retenus.

| Critère du protocole | Résultat |
|---|---|
| Couverture ≥80 % | **Échec : 45,83 %** |
| Erreur moyenne acceptée ≤12/255 | Passe |
| Gain sur le voisin connu ≥10 % | Passe |
| Propositions fortement dégradées ≤5 % | Passe sur ce petit lot |
| Temps p95 ≤1 s | Passe |
| Conservation des pixels connus | Passe |

**Le critère global reste refusé.** Les seuils ont été conservés. Zéro erreur
forte parmi 11 propositions ne démontre pas une fiabilité générale : les 24 cas
proviennent seulement de quatre sources photographiques. Aucun résultat terrain,
intervalle de fiabilité, test utilisateur ou preuve sur de grandes images n'est
déduit de cet essai.

Le détourage, inchangé, retrouve les deux compositions contrôlées (IoU 1,0) et
s'abstient au faible contraste. Cela reste distinct d'une segmentation photographique.

## Vérification du paquet

Le candidat intégré du 16 septembre passe **236 tests** et le contrôle Ruff.
La wheel et l'archive source se construisent ; l'installation hors dépôt est
vérifiée dans un environnement sans Pillow, puis avec Pillow. Le chargement des
28 ressources JSON natives, des 11 réseaux historiques et des 42 entrées du
registre est contrôlé. Les fonctionnalités ajoutées en parallèle sont présentes
dans cet instantané, sans être créditées d'une validation photographique.

L'empreinte du contrôle de remplissage embarqué correspond exactement à celle
du moteur évalué. Les poids évalués sont inchangés. Les preuves de construction
et d'installation sont dans `C:\BelliumAI\stabilisation-2026-09-16`.
Ces succès techniques sont distincts de l'échec du critère de couverture.

## Reproduction et suite

Les preuves sont conservées dans `C:\BelliumAI\evaluation-2026-09-16\visual-v2` :
`report.json`, `protocol.json`, `engine-manifest.json`, `RAPPORT.md` et les vues
de diagnostic. La copie précédente est dans le répertoire `baseline` voisin.

```sh
python scripts/evaluate_visual_quality.py --manifest benchmarks/manifests/visual-quality-v2.json --cache <cache> --output <nouveau-dossier> --baseline-module <copie-epinglee> --fetch
```

Omettre `--fetch` pour rejouer hors ligne. Le programme refuse un hash incorrect
ou l'écrasement d'un rapport existant. La réussite des tests de fonctionnement
est distincte du refus du critère de qualité.

La prochaine étape vise une meilleure couverture sans réintroduire les erreurs
fortes : comparer une sélection locale des méthodes simples, caractériser les
zones réellement prévisibles, puis fixer un troisième lot inédit avant toute
nouvelle évaluation. Les données v2 ne doivent pas devenir un test « inédit »
après avoir guidé une modification. Aucun passage en mode actif ni lancement
automatique d'un modèle lourd n'a été ajouté.
