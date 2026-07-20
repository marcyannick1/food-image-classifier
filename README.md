# 🍔 Food Image Classifier

Projet de Deep Learning visant à reconnaître automatiquement le type d'un plat à partir d'une image.

---

# Objectif

Développer une application capable de classifier une image de nourriture parmi les 101 catégories du dataset Food-101.

L'objectif est de produire un modèle performant tout en restant suffisamment léger pour être intégré dans une application web.

---

# Architecture du projet

```
                 Image
                   │
                   ▼
        Prétraitement (Resize + Normalize)
                   │
                   ▼
          EfficientNetB0 pré-entraîné
                   │
        (Transfer Learning ImageNet)
                   │
                   ▼
      Global Average Pooling
                   │
                   ▼
          Couche Dense (101 classes)
                   │
                   ▼
             Softmax
                   │
                   ▼
          Classe prédite
```

## Pourquoi EfficientNetB0 ?

Nous avons choisi **EfficientNetB0** pour plusieurs raisons :

- architecture légère et rapide, bon compromis précision/coût de calcul
- excellentes performances en classification d'images pour sa taille
- modèle pré-entraîné sur ImageNet, disponible directement via `tf.keras.applications`
- adapté au transfert d'apprentissage (`include_top=False`)
- facilement déployable dans une WebApp

Même si Food-101 contient plus de 100 000 images, entraîner un CNN entièrement from scratch demanderait beaucoup plus de temps de calcul et risquerait d'être moins performant dans le temps imparti.

Le transfert d'apprentissage permet d'obtenir rapidement de très bons résultats.

---

# Dataset

## Source

Food-101

https://www.kaggle.com/datasets/dansbecker/food-101

## Contenu

- 101 catégories de plats
- environ 1 000 images par classe
- environ 101 000 images

## Répartition

Le dataset fournit déjà un découpage officiel :

- Train : 75 750 images
- Validation (= test officiel) : 25 250 images

Une partie du train est utilisée comme validation (80/20) :

- Train : 60 %
- Validation : 20 %
- Test : 20 %

Le split sera effectué de manière stratifiée afin de conserver une distribution équilibrée entre les classes.

---

# Prétraitement

Les images subissent les transformations suivantes (`src/dataset.py`) :

- Resize 224×224
- Normalisation des pixels ([0, 1])

Pendant l'entraînement (`src/preprocessing.py`) :

- Random Horizontal Flip
- Random Rotation
- Random Zoom
- Random Contrast
- Random Translation

Ces augmentations permettent de limiter l'overfitting.

---

# Plan d'entraînement

## Plateforme

Google Colab (GPU T4)

---

## Framework

- Python
- Tensorflow
- Keras

---

## Optimizer

Adam

Learning rate initial :

```
0.001
```

Ajusté automatiquement en cours d'entraînement via `ReduceLROnPlateau`.

---

## Loss Function

Sparse Categorical Crossentropy

Adaptée à une classification multi-classe avec labels entiers (non one-hot).

---

## Métrique principale

Accuracy Top-1

Éventuellement :

- Top-5 Accuracy
- Matrice de confusion
- Precision / Recall par classe

---

## Nombre d'epochs

Premier objectif :

```
20 epochs
```

Avec `EarlyStopping` pour arrêter automatiquement si la validation ne progresse plus.

---

## Sauvegarde

Le meilleur modèle est sauvegardé automatiquement via `ModelCheckpoint`, selon :

- meilleure validation accuracy

Suivi des métriques d'entraînement via **TensorBoard**.

---

# Organisation du projet

```
food-image-classifier/

├── data/
│   ├── images/
│   ├── meta/
│   ├── license_agreement.txt
│   └── README.txt
│
├── models/
│   └── README.md
│
├── notebooks/
│   ├── checkpoints/
│   ├── logs/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_training.ipynb
│   └── README.md
│
├── src/
│   ├── dataset.py
│   ├── preprocessing.py
│   └── train.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

# Web Application

Une interface web permettra :

- charger une image
- lancer la prédiction
- afficher :

```
Image

↓

Classe prédite

↓

Confiance (%)
```

Une évolution possible serait d'afficher les 3 meilleures prédictions.

---

# Répartition des rôles

Les responsabilités sont réparties de la manière suivante :

| Tâche | Responsable |
|--------|-------------|
| Préparation du dataset | À définir |
| Développement du modèle | À définir |
| Entraînement et expérimentation | À définir |
| Développement de la WebApp | À définir |
| Documentation et soutenance | À définir |

Les rôles pourront évoluer au cours du projet selon les besoins.

---

# Questions ouvertes

Plusieurs choix techniques restent à valider après les premiers essais :

- comparer MobileNetV2 avec ResNet18 ou EfficientNet-B0
- décider si toutes les couches seront fine-tunées ou seulement le classifieur final
- ajuster les hyperparamètres (learning rate, batch size, scheduler)
- déterminer les meilleures augmentations de données
- choisir le seuil de confiance à afficher dans la WebApp

Ces décisions seront prises après les premiers entraînements.

---

# Planning

## Étape 1

- Préparation du dataset
- Vérification des classes
- Création du pipeline `tf.data`

## Étape 2

- Data augmentation

## Étape 3

- Implémentation du modèle (EfficientNetB0)

## Étape 4

- Entraînement

## Étape 5

- Évaluation

## Étape 6

- Développement de la WebApp

## Étape 7

- Préparation de la démonstration