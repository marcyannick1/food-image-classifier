import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from src.dataset import load_datasets
from src.model import build_model


def train(dataset_path="data", epochs=10, batch_size=32, augment=False):
    train_ds, val_ds, test_ds, classes = load_datasets(
        dataset_path,
        batch_size=batch_size,
    )

    model = build_model(augment=augment)

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
    )

    return model, history, test_ds, classes


if __name__ == "__main__":
    train()
