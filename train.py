import os
from argparse import ArgumentParser

import keras
from keras import backend as K

from keras.datasets import cifar10

from vit import VisionTransformer
from AdamW import AdamW


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--logdir", default="logs")
    parser.add_argument("--image-size", default=32, type=int)
    parser.add_argument("--patch-size", default=4, type=int)
    parser.add_argument("--num-layers", default=8, type=int)
    parser.add_argument("--d-model", default=64, type=int)
    parser.add_argument("--num-heads", default=4, type=int)
    parser.add_argument("--mlp-dim", default=128, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    args = parser.parse_args(args=[])


    # Load the dataset
    (X_train, y_train), (X_val, y_val) = cifar10.load_data()

    model = VisionTransformer(
        image_size=args.image_size,
        patch_size=args.patch_size,
        num_layers=args.num_layers,
        num_classes=10,
        d_model=args.d_model,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
        channels=3,
        dropout=0.1,
    )
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        ),
        optimizer=AdamW(lr=args.lr, weight_decay=args.weight_decay),
        metrics=["accuracy"],
    )

    early_stop = keras.callbacks.EarlyStopping(patience=10, monitor='loss'),
    mcp = keras.callbacks.ModelCheckpoint(filepath='weights/best.weights',
        monitor='loss', verbose=1, save_best_only=True, save_weights_only=True)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='loss', factor=0.1, patience=3, verbose=0, mode='auto',
        min_delta=0.0001, cooldown=0, min_lr=0)    

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        callbacks=[*early_stop, mcp, reduce_lr],
    )
