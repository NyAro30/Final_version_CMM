# Récapitulatif des modifications

Ce document décrit les changements apportés pour rendre le mandatory
exécutable, réduire la taille de `src/schema.py` et vérifier les résultats.
Le README est volontairement laissé hors de ce périmètre.

## Résultat en bref

- Qwen est chargé via le SDK fourni.
- `src/schema.py` reste sous la limite visée de 300 lignes.
- Le choix de fonction et l'extraction des arguments passent par le modèle ;
  aucune valeur n'est préremplie par des regex ou des heuristiques.
- Une erreur sur un prompt conserve et écrit les résultats des prompts
  réussis, puis renvoie un code de sortie non nul.

## Changements par fichier

### Dépendances et tâches

- `pyproject.toml` déclare les dépendances nécessaires pour charger Qwen
  avec le SDK : PyTorch, Transformers, Hugging Face Hub et Accelerate.
- `uv.lock` a été régénéré pour correspondre au manifeste et à l'index CPU
  de PyTorch.
- `.flake8` exclut le virtualenv et le code du SDK fourni des vérifications
  du projet.
- `Makefile` contient les cibles obligatoires `install`, `run`, `debug`,
  `clean` et `lint`, avec les commandes flake8/mypy demandées. Il ajoute
  également `lint-strict` et `test`.

### Entrée du programme et modèles

- `src/__main__.py` valide les fichiers d'entrée, construit le modèle,
  traite chaque prompt, vérifie le résultat avec Pydantic et écrit un JSON
  strict dans le chemin de sortie demandé. Les erreurs de chargement, de
  validation, d'inférence et d'écriture sont signalées avec un code de
  sortie non nul.
- `src/models.py` définit les protocoles des méthodes publiques attendues
  du modèle et limite les types d'arguments acceptés à `number`, `string`
  et `boolean`.
- `src/vocab.py` utilise le protocole public du modèle pour charger les
  textes décodés du vocabulaire.

### Décodage et extraction des arguments

- `src/schema.py` conserve la sélection de fonction par le modèle : à
  chaque étape, les logits sont restreints aux tokens qui peuvent compléter
  un nom de fonction autorisé.
- Le même module contraint le décodage des valeurs aux préfixes
  compatibles avec un nombre, une chaîne ou un booléen JSON, puis vérifie
  la valeur complète et son type.
- Toutes les valeurs d'arguments sont générées par le modèle, sans extraction
  préalable. `src/prompts.py` décrit la fonction, la position et le rôle de
  chaque argument ; les prompts précisent le dialecte regex Python et
  donnent des exemples génériques.
- Le parcours des tokens s'arrête dès le premier token valide, sans construire
  une liste complète de candidats décodés.
- Les nombres sont conservés comme des nombres JSON flottants dans les
  résultats.
- Les chaînes de demande utilisent le format de conversation Qwen attendu.

### Tests

- `tests/test_schema.py` couvre les préfixes JSON, le refus des types de
  paramètres inconnus, l'ignorance d'une définition de fonction invalide,
  l'arrêt précoce du parcours des tokens et la génération contrainte avec un
  faux modèle déterministe.
- Le fichier de sortie des exemples est ignoré par Git et n'est pas conservé
  comme livrable.

## Vérifications

Les contrôles suivants passent après le retrait des heuristiques :

```text
make lint
make lint-strict
make test
uv lock --check
```

## Limite importante du sujet

Le sujet demande à la fois Qwen via le SDK fourni et interdit PyTorch,
Transformers et Hugging Face, qui sont nécessaires à ce SDK. Pour rendre le
programme exécutable, les dépendances du SDK sont incluses. La contradiction
du sujet n'est donc pas résolue : cette exception doit être confirmée avec
l'encadrant avant l'évaluation.

La conformité sémantique du modèle doit être mesurée sur les entrées
d'évaluation ; les prompts et le décodage contraint ne garantissent pas à eux
seuls que la valeur choisie est correcte.
