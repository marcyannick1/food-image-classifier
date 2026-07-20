import tensorflow as tf
import tensorflow_datasets as tfds


# =====================================================
# PREPROCESSING
# =====================================================

def preprocess(image, label, image_size=(224, 224)):
    """
    Redimensionne et normalise une image.
    """
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


# =====================================================
# TF.DATASET
# =====================================================

def build_dataset(
    ds,
    image_size=(224, 224),
    batch_size=32,
    shuffle=True,
    shuffle_buffer=1000,
    cache=False,
    cache_path="",
):
    """
    Applique le preprocessing + pipeline (cache/shuffle/batch/prefetch)
    à un tf.data.Dataset issu de tfds.load().
    """

    ds = ds.map(
        lambda image, label: preprocess(image, label, image_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if cache:
        # cache_path="" -> cache en RAM
        # cache_path="/chemin/vers/fichier" -> cache sur disque
        ds = ds.cache(cache_path)

    if shuffle:
        ds = ds.shuffle(
            buffer_size=shuffle_buffer,
            reshuffle_each_iteration=True,
        )

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


# =====================================================
# LOAD DATASETS
# =====================================================

def load_datasets(
    dataset_path=None,
    image_size=(224, 224),
    batch_size=32,
    validation_size=0.2,
):
    """
    Charge Food-101 via tensorflow_datasets.

    dataset_path : ignoré, gardé pour compatibilité avec les appels
                   existants (ex. load_datasets("../data")).
                   tfds télécharge et stocke les données dans
                   ~/tensorflow_datasets/ par défaut.

    Retourne :
        train_ds
        val_ds
        test_ds
        classes
    """

    # tfds ne fournit que "train" et "validation" (= test officiel).
    # On découpe "train" en train/val selon validation_size.
    val_pct = int(validation_size * 100)
    train_split = f"train[{val_pct}%:]"
    val_split = f"train[:{val_pct}%]"

    (raw_train, raw_val, raw_test), info = tfds.load(
        "food101",
        split=[train_split, val_split, "validation"],
        as_supervised=True,
        with_info=True,
    )

    classes = info.features["label"].names

    train_ds = build_dataset(
        raw_train,
        image_size,
        batch_size,
        shuffle=True,
    )

    val_ds = build_dataset(
        raw_val,
        image_size,
        batch_size,
        shuffle=False,
    )

    test_ds = build_dataset(
        raw_test,
        image_size,
        batch_size,
        shuffle=False,
    )

    return (
        train_ds,
        val_ds,
        test_ds,
        classes,
    )