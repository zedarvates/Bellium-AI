# Contrat des estimations de conséquences — correction du 17 septembre 2026

L'architecture Bellium distingue une proposition de l'autorité d'exécution.
La roadmap P2 réserve cette autorité à la politique déterministe et au système
appelant ; les fiches des conséquences déclarent le spécialiste consultatif et
`executes=False`. Aucune estimation « safe » ne doit donc dispenser cet appelant
de sa propre confirmation.

L'audit des appels n'a trouvé que les tests comme consommateurs de
`predict_consequence`. Deux scénarios anciens utilisaient des contraintes vides
pour interroger les étages appris ; l'un attendait `requires_confirmation=False`.
La suite de limites demandait au contraire l'abstention pour des contraintes
absentes et conservait la confirmation. Cette contradiction est résolue en faveur
de la limite d'autorité documentée, avec la migration explicite suivante.

- `constraints` doit être un objet quand il est fourni ; `None`, `False`, zéro,
  chaîne vide et liste ne sont pas des objets de contraintes valides.
- Les trois champs `irreversible`, `has_backup`, `dry_run_available` doivent être
  explicitement booléens pour consulter les modèles. Un champ absent reste inconnu.
- Un veto conservateur demeure prioritaire si l'état ou les déclarations signalent
  une action irréversible sans protection. Sinon, absence ou contradiction entraîne
  une abstention avant les étages appris.
- Les états utilisent les mêmes contrôles dans les trois étages. Les ratios doivent
  être numériques, finis et compris entre 0 et 1, sans écrêtage. Les champs drapeaux
  acceptent également les booléens, normalisés en 0/1 ; aucun texte libre n'est converti.
- Le champ `verified` d'un précédent est booléen, jamais interprété par sa vérité
  Python. Les identifiants sont uniques et la mémoire injectée suit le même contrôle
  que le fichier chargé.
- `requires_confirmation=True`, `executes=False`, `certified=False` sont conservés
  sur toutes les réponses hybrides, y compris « safe ».

Les scénarios positifs doivent maintenant fournir les trois contraintes connues
et cohérentes. Les tests de limites ne sont pas assouplis. La règle de risque,
les poids, les résultats consultatifs et les mémoires de fixtures restent inchangés.
