from Logic.ReaderL import ReaderL
from Logic.Neuron import Neuron
import tensorflow as tf
import matplotlib.pyplot as plt


class Layer:
    def __init__(self, num_neurons, num_inputs):
        self.neurons = [Neuron(num_inputs) for _ in range(num_neurons)]

    def forward(self, inputs):
        return [neuron.calculate_output(inputs) for neuron in self.neurons]

    def backward_propagation(self, target_or_next_layer, learning_rate, is_output_layer=False):
        if is_output_layer:
            for i, neuron in enumerate(self.neurons):
                error = target_or_next_layer[i] - neuron.output
                neuron.delta = error * neuron.output * (1 - neuron.output)
        else:
            for i, neuron in enumerate(self.neurons):
                neuron.delta = sum(next_neuron.weights[i] * next_neuron.delta for next_neuron in target_or_next_layer.neurons)
                neuron.delta *= neuron.output * (1 - neuron.output)
        
        for neuron in self.neurons:
            for j in range(len(neuron.weights)):
                neuron.weights[j] += learning_rate * neuron.delta * neuron.inputs[j]
            neuron.bias += learning_rate * neuron.delta


class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layers = [Layer(layer_sizes[i], layer_sizes[i-1]) for i in range(1, len(layer_sizes))]
        self.accuracy_history = []
        self.loss_history = []
        self.log_dir = "logs"
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)

    def forward_propagation(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backpropagate(self, expected_output, learning_rate):
        self.layers[-1].backward_propagation(expected_output, learning_rate, is_output_layer=True)
        for i in range(len(self.layers) - 2, -1, -1):
            self.layers[i].backward_propagation(self.layers[i + 1], learning_rate)

    def train(self, data, labels, epochs, learning_rate):
        for epoch in range(epochs):
            correct = 0
            total_loss = 0
            for inputs, expected in zip(data, labels):
                outputs = self.forward_propagation(inputs)
                predicted = outputs.index(max(outputs))
                correct += (predicted == expected)
                
                expected_output = [0] * 10
                expected_output[expected] = 1

                loss = sum((eo - o) ** 2 for eo, o in zip(expected_output, outputs)) / len(outputs)
                total_loss += loss

                self.backpropagate(expected_output, learning_rate)

            accuracy = correct / len(data)
            avg_loss = total_loss / len(data)
            self.accuracy_history.append(accuracy)
            self.loss_history.append(avg_loss)

            with self.summary_writer.as_default():
                tf.summary.scalar("Accuracy", accuracy, step=epoch)
                tf.summary.scalar("Loss", avg_loss, step=epoch)

            print(f"Epoch {epoch+1}: Accuracy = {accuracy * 100:.2f}%, Loss = {avg_loss:.4f}")

    def test(self, data, labels):
        correct = 0
        for inputs, expected in zip(data, labels):
            outputs = self.forward_propagation(inputs)
            predicted = outputs.index(max(outputs))
            correct += (predicted == expected)
        return correct / len(data)

    def plot_metrics(self):
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(self.accuracy_history, label="Accuracy", color="blue")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.loss_history, label="Loss", color="red")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.show()


# Example usage:
def main():
    train_data, train_labels = ReaderL.read_file("mnist_train.csv")
    test_data, test_labels = ReaderL.read_file("mnist_test.csv")

    nn = NeuralNetwork([784, 16, 16, 10])
    nn.train(train_data, train_labels, epochs=200, learning_rate=0.1)

    accuracy = nn.test(test_data, test_labels)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    nn.plot_metrics()


if __name__ == "__main__":
    main()
