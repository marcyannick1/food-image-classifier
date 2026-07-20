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
           MobileNetV2 pré-entraîné
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

## Pourquoi MobileNetV2 ?

Nous avons choisi **MobileNetV2** pour plusieurs raisons :

- architecture légère et rapide
- excellentes performances en classification d'images
- modèle pré-entraîné sur ImageNet
- adapté au transfert d'apprentissage
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

Le dataset est organisé sous la forme :

```
images/
    pizza/
    sushi/
    burger/
    ...
meta/
    classes.txt
    labels.txt
    train.txt
    test.txt
```

Chaque image est déjà classée dans son dossier correspondant.

---

## Répartition

Le dataset fournit déjà un découpage officiel :

- Train : 75 750 images
- Test : 25 250 images

Une partie du train sera utilisée comme validation (par exemple 80/20).

Répartition envisagée :

- Train : 60 %
- Validation : 20 %
- Test : 20 %

Le split sera effectué de manière stratifiée afin de conserver une distribution équilibrée entre les classes.

---

# Prétraitement

Les images subiront les transformations suivantes :

- Resize 224×224
- Conversion en Tensor
- Normalisation ImageNet

Pendant l'entraînement :

- Random Horizontal Flip
- Random Rotation
- Random Crop
- Color Jitter (si nécessaire)

Ces augmentations permettront de limiter l'overfitting.

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

---

## Loss Function

CrossEntropyLoss

Adaptée à une classification multi-classe.

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
10 epochs
```

Puis augmentation progressive :

- 20
- 30

si la validation continue de progresser.

---

## Sauvegarde

Le meilleur modèle sera sauvegardé selon :

- meilleure validation accuracy

---

# Organisation du projet

```
food-image-classifier/

├── data/
│── src/
├──────dataset.py
├──────model.py
├──────train.py
├──────predict.py
├── models/
│
├── notebooks/
│
├── webapp/
│
├── requirements.txt
│
└── README.md
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
- Création du DataLoader

## Étape 2

- Implémentation du modèle

## Étape 3

- Entraînement

## Étape 4

- Évaluation

## Étape 5

- Développement de la WebApp

## Étape 6

- Préparation de la démonstration