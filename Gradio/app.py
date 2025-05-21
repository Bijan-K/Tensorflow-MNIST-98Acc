import os
import gradio as gr
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from efficientnet.tfkeras import EfficientNetB0

model = keras.Sequential(
    [
        layers.Input(shape=(32, 32, 1)),
        layers.experimental.preprocessing.Rescaling(1.0 / 255),
        EfficientNetB0(include_top=False, weights=None, input_shape=(32, 32, 1)),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(10),
    ]
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model = load_model("my_model.h5")


def classify_image(image):
    # Preprocess the image
    image_gray = tf.image.rgb_to_grayscale(image)
    image_tensor = tf.convert_to_tensor(image_gray)

    # Resize the image to 28x28.
    image_tensor = tf.image.resize(image_tensor, (32, 32))

    # Cast the data to float32.
    image_tensor = tf.cast(image_tensor, tf.float32)

    # Add a batch dimension.
    image_tensor = tf.expand_dims(image_tensor, 0)

    # Normalize the data.
    image_tensor = image_tensor / 255.0

    # Get the prediction.
    prediction = model.predict(image_tensor)

    # Convert the prediction to a string label.
    prediction_label = str(prediction.argmax())

    return prediction_label


title = "MNIST Model 98%acc"
description = "Model trained on MNIST dataset using efficientnet to classify MNIST images with 98% accuracy"
article = "for source code you can visit [my github](https://github.com/Bijan-K/Tensorflow-MNIST-98Acc.git) (gradio + training code)."

example_list = [["examples/" + example] for example in os.listdir("examples")]

interface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3, label="Predictions"),
    examples=example_list,
    title=title,
    description=description,
    article=article,
)


interface.launch()
