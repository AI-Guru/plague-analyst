import sir_dataset
import keras
from keras import models
from keras import layers
from keras import optimizers
import numpy as np
import matplotlib.pyplot as plt

epochs = 50
batch_size = 128

def main():
    # TODO Ensure dataset.

    print("Loading dataset...")
    dataset = sir_dataset.load_dataset("dataset-counts-10000.p")
    sir_dataset.print_dataset_statistics(dataset)

    (train_input, train_output), (validate_input, validate_output), (test_input, test_output) = dataset

    # Normalize all input data.
    population_size = np.sum(train_input[0,0])
    train_input = train_input / population_size
    validate_input = validate_input / population_size
    test_input = test_input / population_size

    # Normalize all output data.
    minimum = np.amin(train_output, axis=0)
    maximum = np.amax(train_output, axis=0)
    difference = maximum - minimum
    train_output = (train_output - minimum) / difference
    validate_output = (validate_output - minimum) / difference
    test_output = (test_output - minimum) / difference


    model_types = ["dense", "lstm", "deeplstm", "gru", "deepgru"]
    for model_type in model_types:

        # Create the model.
        model = create_model(model_type, train_input.shape, train_output.shape)

        # Train the model.
        history = model.fit(
            train_input, train_output,
            validation_data=(validate_input, validate_output),
            epochs=epochs,
            batch_size=batch_size
        )

        # Plot the history.
        plot_history(history, model_type)

        # Evaluate the model against the test-data.
        evaluation_result = model.evaluate(test_input, test_output)
        print(evaluation_result)


def create_model(model_type, input_shape, output_shape):

    model = models.Sequential()

    if model_type == "dense":
        model.add(layers.Flatten(input_shape=(input_shape[1], input_shape[2])))
        model.add(layers.Dense(300, activation="relu"))
        model.add(layers.Dense(output_shape[1], activation="sigmoid"))
    elif model_type == "lstm":
        model.add(layers.LSTM(30, input_shape=(input_shape[1], input_shape[2])))
        model.add(layers.Dense(output_shape[1], activation="sigmoid"))
    elif model_type == "deeplstm":
        model.add(layers.LSTM(30, input_shape=(input_shape[1], input_shape[2]), return_sequences=True))
        model.add(layers.LSTM(20, return_sequences=True))
        model.add(layers.LSTM(10))
        model.add(layers.Dense(output_shape[1], activation="sigmoid"))
    elif model_type == "gru":
        model.add(layers.GRU(10, input_shape=(input_shape[1], input_shape[2])))
        model.add(layers.Dense(output_shape[1], activation="sigmoid"))
    elif model_type == "deepgru":
        model.add(layers.GRU(30, input_shape=(input_shape[1], input_shape[2]), return_sequences=True))
        model.add(layers.GRU(20, return_sequences=True))
        model.add(layers.GRU(10))
        model.add(layers.Dense(output_shape[1], activation="sigmoid"))
    else:
        raise Exception("Unknown model type:", model_type)

    model.summary()

    model.compile(
        loss="mse",
        optimizer=optimizers.RMSprop(lr=0.01),
        metrics=["accuracy"]
    )

    return model


def plot_history(history, prefix):
    """ Plots the history. """

    # Render the accuracy.
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(prefix + "history_accuracy.png")
    plt.clf()

    # Render the loss.
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(prefix + "history_loss.png")
    plt.clf()
    


if __name__ == "__main__":
    main()
