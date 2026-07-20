import os
import datetime
import tensorflow as tf
from tensorflow import keras


# =====================================================
# CALLBACKS
# =====================================================

def get_callbacks(
    checkpoint_dir="checkpoints",
    log_dir="logs",
    patience_early_stopping=5,
    patience_reduce_lr=3,
):
    """
    Construit les callbacks utilisés pendant l'entraînement.
    """

    os.makedirs(checkpoint_dir, exist_ok=True)

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_dir = os.path.join(log_dir, run_id)

    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, "best_model.keras"),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )

    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience_early_stopping,
        restore_best_weights=True,
        verbose=1,
    )

    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=run_log_dir,
        histogram_freq=1,
    )

    reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=patience_reduce_lr,
        min_lr=1e-7,
        verbose=1,
    )

    return [
        checkpoint_callback,
        early_stopping_callback,
        tensorboard_callback,
        reduce_lr_callback,
    ]


# =====================================================
# MODELE PAR DEFAUT (transfer learning)
# =====================================================

def build_model(num_classes, image_size=(224, 224), fine_tune=False):
    """
    Modèle de base : EfficientNetB0 pré-entraîné sur ImageNet,
    tête de classification adaptée au nombre de classes.

    Si tu as déjà un modèle défini dans un autre fichier
    (ex. src/model.py issu de la Phase 4), utilise-le à la
    place de cette fonction.
    """

    base_model = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*image_size, 3),
    )
    base_model.trainable = fine_tune

    inputs = keras.Input(shape=(*image_size, 3))
    x = keras.layers.Rescaling(255.0)(inputs)
    x = keras.applications.efficientnet.preprocess_input(x)
    x = base_model(x, training=fine_tune)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)

    return model


# =====================================================
# ENTRAINEMENT
# =====================================================

def train_model(
    model,
    train_ds,
    val_ds,
    epochs=20,
    learning_rate=1e-3,
    checkpoint_dir="checkpoints",
    log_dir="logs",
):
    """
    Compile et entraîne un modèle avec les callbacks standards.

    Returns
    -------
    history : keras.callbacks.History
    """

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = get_callbacks(
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    return history