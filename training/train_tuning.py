import keras_tuner as kt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def model_builder(hp):
    """
    Builds a CNN model for hyperparameter tuning.
    """
    model = Sequential()
    model.add(Conv2D(hp.Int("filters", 32, 128, step=32), kernel_size=3, activation="relu", input_shape=(128, 128, 1)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(hp.Int("units", 64, 256, step=64), activation="relu"))
    model.add(Dropout(hp.Float("dropout", 0.2, 0.5, step=0.1)))
    model.add(Dense(5, activation="softmax"))

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def run_hyperparameter_tuning(x_train, y_train, x_val, y_val):
    """
    Runs hyperparameter tuning for the model using Keras Tuner.

    Args:
        x_train (numpy.ndarray): Training data.
        y_train (numpy.ndarray): Training labels.
        x_val (numpy.ndarray): Validation data.
        y_val (numpy.ndarray): Validation labels.

    Returns:
        tuner (kt.Hyperband): Keras Tuner object with the best hyperparameters.
    """
    tuner = kt.Hyperband(
        model_builder,
        objective="val_accuracy",
        max_epochs=10,
        factor=3,
        directory="tuner_logs",
        project_name="cnn_tuning"
    )

    print("Starting hyperparameter tuning...")
    tuner.search(x_train, y_train, epochs=10, validation_data=(x_val, y_val))

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"Best hyperparameters: filters={best_hps.get('filters')}, units={best_hps.get('units')}, dropout={best_hps.get('dropout')}")
    return tuner
