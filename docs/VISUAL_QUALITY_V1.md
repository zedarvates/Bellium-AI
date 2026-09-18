# Évaluation visuelle v1 — 15 septembre 2026

Le protocole a été fixé avant les scores : quatre photographies, trois positions
et deux tailles de trous, soit 24 cas. Les images de café, chat, briques et herbe
proviennent de scikit-image 0.25.2, révision
`e8a42ba85aaf5fd9322ef9ca51bc21063b22fcae`, avec SHA-256 et licences documentées
dans `benchmarks/manifests/visual-quality-v1.json`.

**Le remplissage photographique ne passe pas le critère de qualité.**

| Mesure | Résultat | Critère |
|---|---:|---:|
| Couverture | 24/24 | ≥ 80 % |
| Erreur moyenne sur le trou, canaux 0–255 | 13,68 | ≤ 12 |
| Erreur du voisin connu le plus proche | 15,36 | référence |
| Gain moyen face à cette référence | 10,92 % | ≥ 10 % |
| Cas acceptés avec erreur > 25 | 4/24, soit 16,67 % | ≤ 5 % |
| Latence p95 | 0,065 s | ≤ 1 s |
| Pixels connus conservés | tous | tous |

Les scores du remplisseur restent compris entre 0,9165 et 0,95, même sur les cas
dégradés. Le routeur propose le patch sur 22 cas, dont trois dépassent l'erreur 25.
Ces scores ne sont donc pas des probabilités calibrées de reconstruction correcte.

Le détourage retrouve exactement la silhouette de cheval sur deux fonds unis
(IoU 1,0) et s'abstient sur le fond peu contrasté. Ce résultat concerne des
compositions contrôlées, pas une segmentation photographique complexe.

Les réseaux et seuils sont restés inchangés pendant ce premier essai. Ce petit
lot, indépendant des fixtures initiales, sert désormais au diagnostic. Il ne doit
plus être présenté comme inédit pour les corrections ultérieures.

Les preuves originales sont conservées dans
`C:\BelliumAI\evaluation-2026-09-15\visual-v1` : `report.json`, `protocol.json`,
`engine-manifest.json`, les comparaisons et le rapport lisible `RAPPORT.md`.
RAM/VRAM, qualité perçue et utilisation dans un consommateur restent à évaluer.
