import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np


def main():
    # Loading the dataset
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
    output_classes = 10

    # Scale images to the [0, 1] range
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Make sure images have shape (28, 28, 1)
    train_images = np.expand_dims(train_images, -1)
    test_images = np.expand_dims(test_images, -1)

    # Build the model
    model = keras.Sequential([
        keras.Input(shape=(28, 28, 1)),
        keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dense(output_classes, activation="softmax")
    ])

    model.compile(optimizer='adam',
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    ## Train the model
    model.fit(train_images, train_labels, epochs=10, batch_size=128)

    ## Evaluate the trained model
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print('\n Test accuracy:', test_acc)

    print(model.predict(test_images)[10])
    plt.imshow(test_images[10].squeeze(), cmap='gray')
    plt.savefig('image.png')

    print('----')
    model.summary()


if __name__ == "__main__":
    main()
