#!/usr/bin/env python3
"""Minimal TensorFlow training example consuming prepare_dataset.py manifests."""

import argparse
import csv
from pathlib import Path

import tensorflow as tf


CLASSES = ("arch", "tented_arch", "left_loop", "right_loop", "whorl")


def load_rows(csv_path):
    with Path(csv_path).open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No records in {csv_path}")
    return rows


def dataset(rows, size, batch, training):
    paths = [r["image_path"] for r in rows]
    labels = [int(r["label_id"]) for r in rows]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(rows), seed=20261008, reshuffle_each_iteration=True)

    def decode(path, label):
        raw = tf.io.read_file(path)
        image = tf.io.decode_image(raw, channels=1, expand_animations=False)
        image.set_shape([None, None, 1])
        image = tf.image.resize(image, [size, size], antialias=True)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    return ds.map(decode, num_parallel_calls=tf.data.AUTOTUNE).batch(batch).prefetch(tf.data.AUTOTUNE)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifests", type=Path, required=True, help="directory containing train.csv, validation.csv, test.csv")
    p.add_argument("--output", type=Path, default=Path("fingerprint_classifier.keras"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--batch", type=int, default=32)
    args = p.parse_args()
    train_rows = load_rows(args.manifests / "train.csv")
    val_rows = load_rows(args.manifests / "validation.csv")
    test_rows = load_rows(args.manifests / "test.csv")
    train_ds = dataset(train_rows, args.size, args.batch, True)
    val_ds = dataset(val_rows, args.size, args.batch, False)
    test_ds = dataset(test_rows, args.size, args.batch, False)

    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.02, fill_mode="constant"),
        tf.keras.layers.RandomContrast(0.15),
        tf.keras.layers.RandomBrightness(0.10, value_range=(0, 1)),
    ], name="training_augmentation")
    inputs = tf.keras.Input((args.size, args.size, 1))
    x = augmentation(inputs)  # Keras runs random augmentation only with training=True.
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    outputs = tf.keras.layers.Dense(len(CLASSES), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)
    print("Held-out test metrics:", model.evaluate(test_ds, return_dict=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    print("Class order:", list(CLASSES))
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
