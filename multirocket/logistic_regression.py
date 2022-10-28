import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from tensorflow.keras.regularizers import l1_l2


class LogisticRegression:
    def __init__(
            self,
            num_features,
            max_epochs=500,
            minibatch_size=256,
            validation_size=2 ** 11,
            learning_rate=1e-4,
            patience_lr=5,  # 50 minibatches
            patience=10,  # 100 minibatches
    ):
        self.name = "LogisticRegression"

        self.args = {
            "num_features": num_features,
            "validation_size": validation_size,
            "minibatch_size": minibatch_size,
            "lr": learning_rate,
            "max_epochs": max_epochs,
            "patience_lr": patience_lr,
            "patience": patience,
        }
        self.model = None
        self.num_classes = None
        self.classes = None

    def fit(self, x_train, y_train):
        training_size = x_train.shape[0]

        args = self.args

        self.scaler = StandardScaler()
        x_train = self.scaler.fit_transform(x_train)

        self.classes = np.unique(y_train)
        self.num_classes = len(self.classes)

        self.enc = OneHotEncoder()
        if self.num_classes > 2:
            y_train = self.enc.fit_transform(y_train.reshape(-1, 1)).toarray()

        # -- model -----------------------------------------------------------------
        out_dims = self.num_classes if self.num_classes > 2 else 1
        out_activation = "softmax" if self.num_classes > 2 else "sigmoid"
        input_layer = tf.keras.layers.Input((x_train.shape[1],))
        output_layer = tf.keras.layers.Dense(
            out_dims,
            activation=out_activation
        )(input_layer)
        model = tf.keras.models.Model(
            inputs=input_layer,
            outputs=output_layer,
            name=self.name
        )
        model.summary()
        # Instantiate an optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=args["lr"])
        # Instantiate a loss function
        loss_fn = tf.keras.losses.CategoricalCrossentropy() if self.num_classes > 2 else tf.keras.losses.BinaryCrossentropy()

        metrics = ["accuracy"]

        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=metrics
        )

        # -- validation data -------------------------------------------------------
        # args["validation_size"] = np.minimum(args["validation_size"], int(0.3 * training_size))
        if args["validation_size"] < training_size:
            x_training, x_validation, y_training, y_validation = train_test_split(
                x_train, y_train,
                test_size=args["validation_size"],
                stratify=y_train
            )

            callbacks = [
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5, min_lr=1e-8,
                    patience=args["patience_lr"]
                ),
            ]

            train_history = model.fit(
                x_training, y_training,
                validation_data=(x_validation, y_validation),
                epochs=args["max_epochs"],
                callbacks=callbacks,
            )
        else:
            callbacks = [
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="loss",
                    factor=0.5, min_lr=1e-8,
                    patience=args["patience_lr"]
                ),
            ]

            train_history = model.fit(
                x_train, y_train,
                epochs=args["max_epochs"],
                callbacks=callbacks,
            )
        self.model = model

    def predict(self, x):
        x = self.scaler.transform(x)

        yhat = self.model.predict(x)
        if self.num_classes > 2:
            yhat = self.classes[np.argmax(yhat, axis=1)]
        else:
            yhat = np.round(yhat)

        return yhat
