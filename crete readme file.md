
***

# Digit Classification using CNN (MNIST)

This project implements a Convolutional Neural Network (CNN) in Keras/TensorFlow to classify handwritten digits from the MNIST dataset.[^1]

## Project Overview

- Uses the MNIST dataset of 28x28 grayscale images of digits 0–9.[^1]
- Builds and trains a CNN model to classify images into 10 classes.[^1]
- Saves the trained model to an HDF5 file named `mnist.h5`.[^1]


## Requirements

The notebook installs and uses the following main dependencies.[^1]

- Python 3
- numpy
- tensorflow
- keras
- pillow

You can install them with:

```bash
pip install numpy tensorflow keras pillow
```


## Dataset

The project uses the MNIST dataset loaded directly from Keras.[^1]

```python
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

- Training set: 60,000 images of size 28x28.[^1]
- Test set: 10,000 images of size 28x28.[^1]


## Data Preprocessing

- Reshapes images to `(samples, 28, 28, 1)` to add the channel dimension.[^1]
- Normalizes pixel values to the range 0–1 by dividing by 255.[^1]
- Converts labels to one-hot encoded vectors for 10 classes.[^1]

```python
x_train = x_train.reshape(x_train.shape[^0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[^0], 28, 28, 1)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
```


## Model Architecture

A Sequential CNN model is used with the following layers.[^1]

- Conv2D with 32 filters, kernel size 3x3, ReLU activation
- Conv2D with 64 filters, kernel size 3x3, ReLU activation
- MaxPooling2D with pool size 2x2
- Dropout with rate 0.25
- Flatten
- Dense with 256 units, ReLU activation
- Dropout with rate 0.5
- Dense output layer with 10 units, softmax activation

The model is compiled with categorical crossentropy loss, Adadelta optimizer, and accuracy metric.[^1]

## Training

- Batch size: 128
- Epochs: 10
- Training data: `x_train`, `y_train`
- Validation data: `x_test`, `y_test`

```python
batch_size = 128
epochs = 10

hist = model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(x_test, y_test)
)
```

The training history includes accuracy and loss for both training and validation sets across 10 epochs.[^1]

## Saving the Model

After training, the model is saved as `mnist.h5`.[^1]

```python
model.save('mnist.h5')
```


## How to Run

1. Clone or download the project repository.
2. Open `digitclassification-model.ipynb` in Jupyter Notebook or Google Colab.
3. Run all cells in order to:
    - Install dependencies
    - Load and preprocess data
    - Build, train, and evaluate the model
    - Save the trained model as `mnist.h5`

## Future Improvements

- Add data augmentation for better generalization.
- Experiment with different optimizers and learning rates.
- Visualize training history and sample predictions.

<div align="center">⁂</div>

[^1]: digitclassification-model.ipynb

