import numpy as np
import pandas as pd
import sys

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    return z * (1 - z)

def normalize(data):
    for i in range(data.shape[1]):
        column_data = data[:, i]
        column_mean = np.mean(column_data)
        column_std = np.std(column_data)

        if column_std != 0:
            data[:, i] = (column_data - column_mean) / column_std
        else:
            data[:, i] = 0

    return data



def train_mlp(inputs, outputs, hidden_neurons, epochs, learning_rate, batch_size):
    input_neurons, output_neurons = inputs.shape[1], outputs.shape[1]
    weights_input_hidden = np.random.uniform(size=(input_neurons, hidden_neurons))
    weights_hidden_output = np.random.uniform(size=(hidden_neurons, output_neurons))

    num_samples = inputs.shape[0]
    num_batches = num_samples // batch_size

    for _ in range(epochs):
        # Shuffle the data for each epoch
        indices = np.random.permutation(num_samples)
        shuffled_inputs = inputs[indices]
        shuffled_outputs = outputs[indices]

        for batch in range(num_batches):
            start = batch * batch_size
            end = (batch + 1) * batch_size

            batch_inputs = shuffled_inputs[start:end]
            batch_outputs = shuffled_outputs[start:end]

            hidden_layer_activation = sigmoid(np.dot(batch_inputs, weights_input_hidden))
            output_layer_activation = sigmoid(np.dot(hidden_layer_activation, weights_hidden_output))

            output_error = batch_outputs - output_layer_activation
            output_delta = output_error * sigmoid_derivative(output_layer_activation)

            hidden_error = np.dot(output_delta, weights_hidden_output.T)
            hidden_delta = hidden_error * sigmoid_derivative(hidden_layer_activation)

            weights_hidden_output += learning_rate * np.dot(hidden_layer_activation.T, output_delta)
            weights_input_hidden += learning_rate * np.dot(batch_inputs.T, hidden_delta)

    return weights_input_hidden, weights_hidden_output


def predict(input_data, weights_input_hidden, weights_hidden_output):
    hidden_layer_activation = sigmoid(np.dot(input_data, weights_input_hidden))
    output_layer_activation = sigmoid(np.dot(hidden_layer_activation, weights_hidden_output))
    return np.eye(output_layer_activation.shape[1])[np.argmax(output_layer_activation, axis=1)]


def main(args):
    name, input_features, output_classes, hidden_neurons, epochs, train_file, test_file = args
    input_features = int(input_features)
    hidden_neurons = int(hidden_neurons)
    epochs = int(epochs)
    train_data = pd.read_csv(train_file, header=None).values
    test_data = pd.read_csv(test_file, header=None).values
    train_inputs, train_outputs = train_data[:, :input_features], train_data[:, input_features:]
    test_inputs, test_outputs = test_data[:, :input_features], test_data[:, input_features:]
    train_inputs, test_inputs = normalize(train_inputs), normalize(test_inputs)
    weights_input_hidden, weights_hidden_output = train_mlp(train_inputs, train_outputs, hidden_neurons, epochs, 0.01,
                                                            20)
    predictions = predict(test_inputs, weights_input_hidden, weights_hidden_output)
    correct_predictions = np.sum(np.all(predictions == test_outputs, axis=1))

    print(f'Accuracy: {correct_predictions / test_outputs.shape[0] * 100:.2f}%')

    output_file = f'B00841761.csv'

    np.savetxt(output_file, predictions, delimiter=',', fmt='%d')


if __name__ == '__main__':
    # main(['main.py', 4, 3, 5, 10000, 'train_data.csv', 'test_data.csv'])
    main(sys.argv)
