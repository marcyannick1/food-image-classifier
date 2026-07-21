import tensorflow as tf
import tensorflow_datasets as tfds
from pathlib import Path


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
    dataset_path,
    image_size=(224, 224),
    batch_size=32,
    validation_size=0.2,
):
    dataset_path = Path(dataset_path)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=validation_size,
        subset="training",
        seed=42,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=validation_size,
        subset="validation",
        seed=42,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
    )

    classes = train_ds.class_names

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)

    # Ici on utilise le jeu de validation comme test.
    test_ds = val_ds

    return (
        train_ds,
        val_ds,
        test_ds,
        classes,
    )