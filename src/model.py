import tensorflow as tf
from tensorflow.keras import layers, models


def _build_backbone(image_size, fine_tune, augment):
    """
    Tronc commun aux deux modeles : MobileNetV2 pre-entraine (gele par
    defaut) + augmentation optionnelle + rescaling [0,1] -> [-1,1]
    (dataset.py normalise en [0,1], MobileNetV2 attend du [-1,1]).
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

    input_layers += [
        layers.Rescaling(scale=2.0, offset=-1.0),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
    ]

    return input_layers


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

    backbone = _build_backbone(image_size, fine_tune, augment)

    model = models.Sequential(backbone + [
        layers.Dense(1, activation="sigmoid"),
    ])

    return model


def build_identify_model(image_size=(224, 224), num_classes=101, fine_tune=False, augment=False):
    """
    Modele secondaire : identifie le plat parmi les 101 classes de
    Food-101. Utilise uniquement en aval du modele binaire, quand
    celui-ci repond "not_hot_dog", pour indiquer de quel plat il
    s'agit probablement.
    """

    backbone = _build_backbone(image_size, fine_tune, augment)

    model = models.Sequential(backbone + [
        layers.Dense(num_classes, activation="softmax"),
    ])

    return model


if __name__ == "__main__":
    model = build_model()
    model.summary()
