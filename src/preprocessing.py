import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    RandomFlip,
    RandomRotation,
    RandomZoom,
    RandomContrast,
    RandomTranslation,
)


# =====================================================
# DATA AUGMENTATION
# =====================================================

def get_data_augmentation():
    """
    Retourne un pipeline Keras de data augmentation.

    Appliqué uniquement sur le train set, jamais sur val/test :
    on veut évaluer le modèle sur des images réalistes,
    pas artificiellement transformées.
    """

    data_augmentation = keras.Sequential(
        [
            RandomFlip("horizontal"),
            RandomRotation(0.1),
            RandomZoom(0.1),
            RandomContrast(0.1),
            RandomTranslation(0.1, 0.1),
        ],
        name="data_augmentation",
    )

    return data_augmentation


def augment_dataset(dataset, augmentation_layer=None):
    """
    Applique la data augmentation à un tf.data.Dataset batché
    de la forme (images, labels).

    À appeler uniquement sur train_ds, après build_dataset()
    et avant l'entraînement.
    """

    if augmentation_layer is None:
        augmentation_layer = get_data_augmentation()

    dataset = dataset.map(
        lambda images, labels: (
            augmentation_layer(images, training=True),
            labels,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    return dataset