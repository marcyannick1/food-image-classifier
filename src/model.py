import tensorflow as tf
from tensorflow.keras import layers, models


def build_model(image_size=(224, 224), fine_tune=False, augment=False):
    """
    MobileNetV2 pre-entraine sur ImageNet + tete de classification
    binaire (hot_dog vs not_hot_dog).

    Les couches de base sont gelees par defaut : seule la tete est
    entrainee, ce qui convient a un dataset de cette taille (~1500
    images). fine_tune=True degele le reseau pour un fine-tuning
    complet une fois la tete deja entrainee.

    augment=True ajoute des couches d'augmentation (flip, rotation,
    zoom) qui ne s'appliquent qu'en entrainement (model.fit), pas en
    inference : pas besoin de toucher au pipeline de dataset.py.
    """

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=image_size + (3,),
        include_top=False,
        weights="imagenet",
    )

    base_model.trainable = fine_tune

    input_layers = [layers.Input(shape=image_size + (3,))]

    if augment:
        input_layers += [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]

    model = models.Sequential(input_layers + [
        # dataset.py normalise deja les images en [0, 1] ; MobileNetV2
        # attend du [-1, 1], d'ou ce rescale avant le reseau pre-entraine.
        layers.Rescaling(scale=2.0, offset=-1.0),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ])

    return model


if __name__ == "__main__":
    model = build_model()
    model.summary()
