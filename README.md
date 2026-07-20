# 🌭 Hotdog / Not Hotdog

Projet de Deep Learning visant à déterminer si une image de plat est un hotdog ou non (classification binaire), à partir d'un sous-ensemble du dataset Food-101.

---

# Objectif

Développer une application capable de répondre à une seule question à partir d'une photo : **"est-ce un hotdog ?"**

Le positif (`hot_dog`) vient de la classe `hot_dog` de Food-101 ; le négatif (`not_hot_dog`) est un échantillon équilibré tiré des 100 autres classes.

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
              Dropout (0.2)
                   │
                   ▼
          Couche Dense (1 neurone)
                   │
                   ▼
             Sigmoid
                   │
                   ▼
     Hotdog (>= 0.5) / Not Hotdog
```

Sortie binaire (1 neurone + sigmoid), pas un softmax à 101 classes : le
problème posé est "hotdog ou pas", pas "quel plat parmi 101".

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

Food-101 fournit 101 catégories, mais nous n'en utilisons que 2 :

- `hot_dog` (positif) : toutes les images de cette classe
- `not_hot_dog` (négatif) : un échantillon aléatoire tiré des 100 autres classes, de même taille que le positif, pour garder un dataset équilibré

Le dataset complet est organisé sous la forme :

```
images/
    pizza/
    hot_dog/
    sushi/
    ...
meta/
    classes.txt
    labels.txt
    train.txt
    test.txt
```

`src/dataset.py::load_binary_split()` lit `train.txt`/`test.txt`, isole les chemins `hot_dog`, et pioche aléatoirement (seed fixe) autant de négatifs parmi les autres classes.

---

## Répartition

Food-101 fournit un split officiel train/test (750/250 images par classe). Après filtrage binaire et équilibrage :

- Train : 1 200 images (600 hot_dog + 600 not_hot_dog)
- Validation : 300 images (20 % du train, split stratifié)
- Test : 500 images (250 hot_dog + 250 not_hot_dog)

Le split train/validation est stratifié (`sklearn.model_selection.train_test_split`, `stratify=labels`) afin de garder 50/50 dans les deux sous-ensembles.

---

# Prétraitement

Pipeline `src/dataset.py` (`tf.data`) :

- Resize 224×224
- Normalisation en [0, 1] (`/255.0`), puis rescale en [-1, 1] dans le modèle (voir `src/model.py`), car MobileNetV2 attend cette plage

Augmentation (optionnelle, activée via `augment=True` dans `build_model()`) :

- Random Horizontal Flip
- Random Rotation (±10 %)
- Random Zoom (±10 %)

Implémentée comme des couches Keras appliquées uniquement pendant `model.fit()` (pas en inférence), donc sans toucher au pipeline `tf.data`.

**Résultat de la comparaison** (voir [Résultats](#résultats)) : sur ce dataset équilibré et de petite taille, l'augmentation n'a pas amélioré les performances — la config **sans augmentation** est retenue.

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

Binary Crossentropy

Adaptée à une classification binaire (sortie sigmoid à 1 neurone).

---

## Métrique principale

Accuracy

Baseline aléatoire pour un problème binaire équilibré : 50 %.

Éventuellement :

- Matrice de confusion (2x2)
- Precision / Recall

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

Le meilleur modèle est sauvegardé selon la meilleure validation accuracy, au format `.keras` dans `models/model.keras` (9,6 Mo, sous la limite GitHub de 100 Mo — pas besoin de Drive/Git LFS).

---

# Résultats

Deux configurations entraînées 8 epochs (MobileNetV2 gelé, tête entraînable uniquement), pour comparer l'effet de la data augmentation :

| Configuration | Train acc | Val acc | Val loss | Test acc | Test loss | Temps |
|---|---|---|---|---|---|---|
| **Sans augmentation** (retenue) | 93.0 % | 91.0 % | 0.230 | **94.2 %** | 0.150 | 518 s |
| Avec augmentation (flip/rotation/zoom) | 91.3 % | 90.3 % | 0.247 | 93.4 % | 0.185 | 621 s |

La config **sans augmentation** est légèrement meilleure sur toutes les métriques, et plus rapide. Explication probable : le dataset est petit (1 200 images train) mais équilibré, et seule la tête (1 281 paramètres) est entraînée sur des features déjà génériques (ImageNet) — l'augmentation ajoute du bruit sans apporter de diversité utile ici. C'est ce modèle qui est sauvegardé dans `models/model.keras`.

## Jalon qualité

**Notre modèle bat-il le baseline aléatoire ?**
Oui. Pour un problème binaire équilibré, le baseline aléatoire est à 50 %. Notre modèle atteint 94.2 % d'accuracy sur le test, très largement au-dessus.

**Est-ce que la loss de validation est inférieure à la loss d'entraînement ?**
Non, légèrement l'inverse (train loss 0.230 vs val loss finale du même ordre, écart train/val acc de 2 points). Ce n'est pas de l'overfitting significatif : avec seulement 1 281 paramètres entraînables (tête) sur une base gelée pré-entraînée, le modèle a peu de marge pour sur-apprendre le train. L'écart est resté stable sur les 8 epochs, sans divergence.

**Quelle configuration a produit le meilleur résultat ? Pourquoi ?**
Sans augmentation. Sur un petit dataset déjà équilibré avec une base gelée, l'augmentation n'apporte pas de bénéfice de généralisation ici — elle ralentit l'entraînement (+20 %) sans améliorer les métriques. L'augmentation serait plus utile avec un fine-tuning complet (base dégelée) ou un dataset plus petit/déséquilibré.

---

# Organisation du projet

```
food-image-classifier/

├── data/                      # dataset Food-101 (non versionne, voir .gitignore)
├── src/
│   ├── dataset.py             # chargement + split binaire hot_dog/not_hot_dog
│   ├── model.py                # MobileNetV2 transfer learning (build_model)
│   └── train.py                # compile + fit (train)
├── models/
│   └── model.keras            # modele entraine (meilleure config)
├── notebooks/
│   └── 01_dataset_exploration.ipynb
├── webapp/
│   └── app.py                  # WebApp Streamlit
├── requirements.txt
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