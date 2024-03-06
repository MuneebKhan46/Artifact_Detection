import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np


def create_model(input_shape, kernel_size, filters, depth, activation, dropout_rate, optimizer):
    model = Sequential()
    for i in range(depth):
        if i == 0:
            model.add(Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, input_shape=input_shape))
        else:
            model.add(Conv2D(filters=filters, kernel_size=kernel_size, activation=activation))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(units=128, activation=activation))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
    model.add(Dense(units=10, activation='softmax'))
    
    if optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = SGD(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = RMSprop(learning_rate=learning_rate)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Parameters for grid search
kernel_sizes = [(3, 3), (5, 5)]
learning_rates = [1e-2, 1e-3]
epochs = [10] # Using a small number for demonstration
batch_sizes = [64, 128]
filters = [32, 64]
depths = [2, 3]
activations = ['relu', 'elu']
dropout_rates = [0.0, 0.25]
optimizers = ['adam', 'sgd']

X_train, X_test, y_train, y_test = load_data()

# Simplified grid search
for kernel_size in kernel_sizes:
    for learning_rate in learning_rates:
        for epoch in epochs:
            for batch_size in batch_sizes:
                for filter in filters:
                    for depth in depths:
                        for activation in activations:
                            for dropout_rate in dropout_rates:
                                for optimizer in optimizers:
                                    model = create_model(input_shape=X_train.shape[1:], kernel_size=kernel_size, filters=filter, depth=depth,
                                                         activation=activation, dropout_rate=dropout_rate, optimizer=optimizer)
                                    model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size, validation_split=0.2, verbose=2)
                                    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
                                    print(f"Accuracy: {accuracy:.4f} | Params: ks={kernel_size}, lr={learning_rate}, ep={epoch}, bs={batch_size}, flt={filter}, dp={depth}, act={activation}, dr={dropout_rate}, opt={optimizer}")

# Note: Running this code as-is will take a significant amount of time due to the number of combinations.
