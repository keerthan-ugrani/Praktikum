from keras.applications import ResNet50
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input

class ResNetClassifier:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def create_model(self):
        base_model = ResNet50(weights="imagenet", include_top=False, input_tensor=Input(shape=self.input_shape))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation="relu")(x)
        predictions = Dense(self.num_classes, activation="softmax")(x)
        model = Model(inputs=base_model.input, outputs=predictions)

        # Freeze the base model layers
        for layer in base_model.layers:
            layer.trainable = False

        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model
