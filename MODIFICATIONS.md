# Récapitulatif des modifications

Ce document décrit les changements apportés pour rendre le mandatory
exécutable, réduire la taille de `src/schema.py` et vérifier les résultats.
Le README est volontairement laissé hors de ce périmètre.

## Résultat en bref

- Le code utilise le SDK fourni avec ses dépendances pour que `make run`
  puisse charger Qwen.
- `src/schema.py` reste sous la limite visée de 300 lignes.
- Le choix de fonction et l'extraction des arguments passent par le modèle ;
  aucune valeur n'est préremplie par des regex ou des heuristiques.
- Une erreur sur un prompt conserve et écrit les résultats des prompts
  réussis, puis renvoie un code de sortie non nul.
- Le run complet a produit 11 appels sur 11 sans erreur ; les trois regex
  générées ont aussi été appliquées aux textes sources et ont donné les
  remplacements attendus. Durée mesurée sur CPU : environ 99 secondes.

## Changements par fichier

### Dépendances et tâches

- `pyproject.toml` déclare les dépendances nécessaires au SDK Qwen :
  PyTorch, Transformers, Hugging Face Hub et Accelerate.
- `uv.lock` reflète ce manifeste et utilise l'index CPU de PyTorch.
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
  préalable. `src/prompts.py` décrit la fonction et les rôles des arguments
  dans un prompt de schéma partagé ; les prompts précisent le dialecte regex
  Python et donnent des exemples génériques.
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
- `src/validator.py` ne garde que la validation d'un JSON complet ; les
  anciens validateurs de préfixe par liste de suffixes ont été retirés.
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

Le fichier de sortie temporaire a été vérifié sur les 11 demandes. Les
fonctions et paramètres correspondent aux attentes, et les trois appels de
substitution ont été exécutés avec `re.sub` afin de vérifier le texte produit.
Cette exécution a été faite avant de retirer les dépendances interdites.

## Limite importante du sujet

Le sujet demande à la fois Qwen via le SDK fourni et interdit PyTorch,
Transformers et Hugging Face, qui sont nécessaires à ce SDK. Pour privilégier
le mandatory exécutable, ces dépendances sont déclarées et `make run` peut
charger le modèle. Cette exception à l'interdiction doit être clarifiée avec
l'encadrant avant l'évaluation.

La vérification sémantique des 11 exemples est passée avec le SDK installé :
les sorties et leurs arguments ont été validés, et les trois regex ont été
exécutées sur leurs textes sources.
