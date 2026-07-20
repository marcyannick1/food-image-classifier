import os
import random

import tensorflow as tf
from sklearn.model_selection import train_test_split

# =====================================================
# BINAIRE : hot_dog vs not_hot_dog
# =====================================================

POSITIVE_CLASS = "hot_dog"
BINARY_CLASSES = ["not_hot_dog", "hot_dog"]


def load_binary_split(dataset_path, split="train", negative_ratio=1.0, random_state=42):
    """
    Lit train.txt ou test.txt et construit un dataset binaire :
    label 1 = hot_dog, label 0 = not_hot_dog.

    Les negatifs sont un echantillon aleatoire pioche parmi les 100
    autres classes, de taille negative_ratio * nombre de positifs,
    pour eviter un dataset desequilibre (750 hot_dog vs 75000 autres).
    """

    split_file = os.path.join(
        dataset_path,
        "meta",
        f"{split}.txt"
    )

    positive_paths = []
    negative_paths = []

    with open(split_file, "r") as f:

        for line in f:

            line = line.strip()

            class_name = line.split("/")[0]

            image_path = os.path.join(
                dataset_path,
                "images",
                line + ".jpg"
            )

            if class_name == POSITIVE_CLASS:
                positive_paths.append(image_path)
            else:
                negative_paths.append(image_path)

    n_negative = int(len(positive_paths) * negative_ratio)
    negative_paths = random.Random(random_state).sample(
        negative_paths,
        n_negative
    )

    image_paths = positive_paths + negative_paths
    labels = [1] * len(positive_paths) + [0] * len(negative_paths)

    return image_paths, labels


# =====================================================
# IMAGE LOADING
# =====================================================

def load_image(path, label, image_size=(224, 224)):
    """
    Charge une image et la prépare.
    """

    image = tf.io.read_file(path)

    image = tf.image.decode_jpeg(
        image,
        channels=3
    )

    image = tf.image.resize(
        image,
        image_size
    )

    image = tf.cast(image, tf.float32) / 255.0

    return image, label


# =====================================================
# TF.DATASET
# =====================================================

def split_train_validation(
    image_paths,
    labels,
    validation_size=0.2,
    random_state=42,
):
    """
    Découpe le train en train + validation.
    """

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths,
        labels,
        test_size=validation_size,
        random_state=random_state,
        stratify=labels,
    )

    return (
        train_paths,
        val_paths,
        train_labels,
        val_labels,
    )


def build_dataset(
    image_paths,
    labels,
    image_size=(224, 224),
    batch_size=32,
    shuffle=True,
    cache=True,
    cache_path="",
):
    """
    Construit un tf.data.Dataset à partir de chemins d'images et de labels.

    Ordre du pipeline :
        1. from_tensor_slices  (paths, labels)
        2. map                 (décodage/resize, coûteux)
        3. cache                (évite de redécoder à chaque epoch)
        4. shuffle              (rejoué à chaque epoch, en aval du cache)
        5. batch
        6. prefetch
    """

    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, labels)
    )

    dataset = dataset.map(
        lambda path, label: load_image(
            path,
            label,
            image_size
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if cache:
        # cache_path="" -> cache en RAM
        # cache_path="/chemin/vers/fichier" -> cache sur disque
        dataset = dataset.cache(cache_path)

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=min(len(image_paths), 1000),
            reshuffle_each_iteration=True
        )

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


# =====================================================
# IDENTIFICATION (101 classes) : modele secondaire, utilise
# uniquement quand le binaire dit "not_hot_dog", pour indiquer
# de quel plat il s'agit probablement.
# =====================================================

def load_classes(dataset_path):
    """
    Charge la liste des 101 classes de Food-101.
    """

    classes_file = os.path.join(dataset_path, "meta", "classes.txt")

    with open(classes_file, "r") as f:
        classes = [line.strip() for line in f]

    class_to_idx = {
        class_name: idx
        for idx, class_name in enumerate(classes)
    }

    return classes, class_to_idx


def load_multiclass_split(dataset_path, split="train", images_per_class=None, random_state=42):
    """
    Lit train.txt ou test.txt avec les 101 classes completes.

    images_per_class limite le nombre d'images gardees par classe
    (echantillonnage aleatoire stratifie) pour accelerer l'entrainement
    du modele d'identification, dont le role est secondaire.
    """

    classes, class_to_idx = load_classes(dataset_path)

    split_file = os.path.join(
        dataset_path,
        "meta",
        f"{split}.txt"
    )

    paths_by_class = {class_name: [] for class_name in classes}

    with open(split_file, "r") as f:

        for line in f:

            line = line.strip()

            class_name = line.split("/")[0]

            image_path = os.path.join(
                dataset_path,
                "images",
                line + ".jpg"
            )

            paths_by_class[class_name].append(image_path)

    rng = random.Random(random_state)

    image_paths = []
    labels = []

    for class_name, paths in paths_by_class.items():

        if images_per_class is not None and len(paths) > images_per_class:
            paths = rng.sample(paths, images_per_class)

        image_paths.extend(paths)
        labels.extend([class_to_idx[class_name]] * len(paths))

    return image_paths, labels, classes


def load_identification_datasets(
    dataset_path,
    image_size=(224, 224),
    batch_size=32,
    validation_size=0.2,
    images_per_class=50,
    cache=True,
):
    """
    Charge le dataset multi-classes (101 plats) pour le modele
    d'identification secondaire.

    Retourne :
        train_ds
        val_ds
        test_ds
        classes  (101 noms de plats)
    """

    train_paths, train_labels, classes = load_multiclass_split(
        dataset_path,
        "train",
        images_per_class
    )

    test_images_per_class = None
    if images_per_class is not None:
        test_images_per_class = max(1, images_per_class // 3)

    test_paths, test_labels, _ = load_multiclass_split(
        dataset_path,
        "test",
        test_images_per_class
    )

    (
        train_paths,
        val_paths,
        train_labels,
        val_labels
    ) = split_train_validation(
        train_paths,
        train_labels,
        validation_size
    )

    train_ds = build_dataset(
        train_paths,
        train_labels,
        image_size,
        batch_size,
        shuffle=True,
        cache=cache,
    )

    val_ds = build_dataset(
        val_paths,
        val_labels,
        image_size,
        batch_size,
        shuffle=False,
        cache=cache,
    )

    test_ds = build_dataset(
        test_paths,
        test_labels,
        image_size,
        batch_size,
        shuffle=False,
        cache=cache,
    )

    return (
        train_ds,
        val_ds,
        test_ds,
        classes,
    )


def dataset_sizes(dataset_path, validation_size=0.2, negative_ratio=1.0):
    """
    Nombre d'images par split (train/val/test), sans decoder
    la moindre image.
    """

    train_paths, train_labels = load_binary_split(
        dataset_path,
        "train",
        negative_ratio
    )

    test_paths, _ = load_binary_split(
        dataset_path,
        "test",
        negative_ratio
    )

    train_paths, val_paths, _, _ = split_train_validation(
        train_paths,
        train_labels,
        validation_size
    )

    return len(train_paths), len(val_paths), len(test_paths)


def load_datasets(
    dataset_path,
    image_size=(224, 224),
    batch_size=32,
    validation_size=0.2,
    negative_ratio=1.0,
    cache=True,
):
    """
    Charge le dataset binaire hot_dog / not_hot_dog.

    Retourne :
        train_ds
        val_ds
        test_ds
        classes  (["not_hot_dog", "hot_dog"])
    """

    train_paths, train_labels = load_binary_split(
        dataset_path,
        "train",
        negative_ratio
    )

    test_paths, test_labels = load_binary_split(
        dataset_path,
        "test",
        negative_ratio
    )

    (
        train_paths,
        val_paths,
        train_labels,
        val_labels
    ) = split_train_validation(
        train_paths,
        train_labels,
        validation_size
    )

    train_ds = build_dataset(
        train_paths,
        train_labels,
        image_size,
        batch_size,
        shuffle=True,
        cache=cache,
    )

    val_ds = build_dataset(
        val_paths,
        val_labels,
        image_size,
        batch_size,
        shuffle=False,
        cache=cache,
    )

    test_ds = build_dataset(
        test_paths,
        test_labels,
        image_size,
        batch_size,
        shuffle=False,
        cache=cache,
    )

    return (
        train_ds,
        val_ds,
        test_ds,
        BINARY_CLASSES,
    )
