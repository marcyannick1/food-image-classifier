import os
import tensorflow as tf
from sklearn.model_selection import train_test_split

# =====================================================
# CLASSES
# =====================================================

def load_classes(dataset_path):
    """
    Charge les 101 classes du dataset.

    Returns
    -------
    classes : list[str]
    class_to_idx : dict
    idx_to_class : dict
    """

    classes_file = os.path.join(dataset_path, "meta", "classes.txt")

    with open(classes_file, "r") as f:
        classes = [line.strip() for line in f]

    class_to_idx = {
        class_name: idx
        for idx, class_name in enumerate(classes)
    }

    idx_to_class = {
        idx: class_name
        for idx, class_name in enumerate(classes)
    }

    return classes, class_to_idx, idx_to_class


# =====================================================
# TRAIN / TEST SPLIT
# =====================================================

def load_split(dataset_path, split="train"):
    """
    Lit train.txt ou test.txt.

    Returns
    -------
    image_paths : list
    labels : list
    """

    classes, class_to_idx, _ = load_classes(dataset_path)

    split_file = os.path.join(
        dataset_path,
        "meta",
        f"{split}.txt"
    )

    image_paths = []
    labels = []

    with open(split_file, "r") as f:

        for line in f:

            line = line.strip()

            class_name = line.split("/")[0]

            image_path = os.path.join(
                dataset_path,
                "images",
                line + ".jpg"
            )

            image_paths.append(image_path)
            labels.append(class_to_idx[class_name])

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


def load_datasets(
    dataset_path,
    image_size=(224, 224),
    batch_size=32,
    validation_size=0.2,
    cache=True,
):
    """
    Retourne :
        train_ds
        val_ds
        test_ds
        classes
    """

    classes, _, _ = load_classes(dataset_path)

    train_paths, train_labels = load_split(
        dataset_path,
        "train"
    )

    test_paths, test_labels = load_split(
        dataset_path,
        "test"
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
