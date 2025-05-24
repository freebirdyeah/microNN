import numpy as np
import warnings
from typing import List
from activations import sigmoid, relu, tanh, softmax
from loss import MSE
from sklearn.datasets import load_iris


activation_functions = {
    "sigmoid": sigmoid,
    "relu": relu,
    "tanh": tanh,
    "softmax": softmax
}


loss_functions = {
    "MSE": MSE,
}


class Neuron():
    def __init__(self, weights=0, bias=0):
        self.weights = weights
        self.bias = bias


class Dense:
    def __init__(self, units: int, activation: str, input_shape=None):
        self.units = units
        self.input_shape = input_shape
        self.activation = activation
        self.layer = [Neuron() for _ in range(units)]
        
        if input_shape is not None:
            self._uniform_xavier_init()

    # Xavier-Glorot Uniform Initialization
    def _uniform_xavier_init(self):
            a = np.sqrt(6/(self.units + self.input_shape))
            self.layer_weight_matrix = np.random.uniform(low=-a, high=a, size=(self.units, self.input_shape))

            for i in range(self.units):
                self.layer[i].weights = self.layer_weight_matrix[i].reshape(self.input_shape, 1)

            self.layer_biases = np.zeros((self.units, 1), dtype=np.float64)


    # Check the weights and biases of each Neuron in Dense layer
    def spit_weights(self):
        for i in range(self.units):
            print(f"For {i+1}th Neuron: w = {self.layer[i].weights}, b = {self.layer[i].bias}")

    def forward(self, input: np.ndarray) -> np.ndarray:
        output = np.dot(self.layer_weight_matrix, input) + self.layer_biases
        if self.activation == "sigmoid":

            # Need to store Z, A for each layer
            self.cache = [output, sigmoid(output)]
            return sigmoid(output)
        
        elif self.activation == "relu":
            self.cache = [output, relu(output)]
            return relu(output)
        
        elif self.activation == "tanh":
            self.cache = [output, tanh(output)]
            return tanh(output)
        
        elif self.activation == "softmax":
            self.cache = [output, softmax(output)]
            return softmax(output)
        
        else:
            raise ValueError("Only sigmoid, tanh, softmax and ReLU activation functions are supported for now :(")


# FLAW: the input_shape param will be defined only for the first Dense layer in the Model(List[Dense])
class Model():
    def __init__(self, layers: List[Dense], loss: str, learning_rate: float):
        
        if not layers:
            raise ValueError("Sequential model must contain at least one layer.")

        # Ensure the first layer has an input_shape defined
        if not hasattr(layers[0], 'input_shape') or layers[0].input_shape is None:
            raise ValueError("First Dense layer must define an 'input_shape'.")
        
        warnings.warn("INPUT_SHAPE PARAMETER VALUE PROVIDED FOR ALL DENSE LAYERS EXCEPT THE FIRST WILL BE OVERIDDEN!")

        self.layers = layers
        self.loss = loss
        self.learning_rate = learning_rate
        self._chain_input_shapes()

    # the dimension of the output vector for a layer is the same as the number of neurons in the layer
    def _chain_input_shapes(self):
        if len(self.layers) > 1:
            for i in range(1, len(self.layers)):
                self.layers[i].input_shape = self.layers[i-1].units

                # Hacky, but since input_shape was None, I couldn't init the params for other Dense Layers earlier
                self.layers[i]._uniform_xavier_init()


    def forward_prop(self, input: np.ndarray) -> np.ndarray:
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output
        

    def backward_prop(self, original_input: np.ndarray, prediction: np.ndarray, target: np.ndarray):
        dL_da = loss_functions[self.loss](prediction, target, deriv=True)

        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            activation = activation_functions[layer.activation]

            z = layer.cache[0]
            a = layer.cache[1]

            da_dz = activation(z, deriv=True)
            dL_dz = dL_da*da_dz

            a_prev = self.layers[i-1].cache[1] if i > 0 else original_input
            dL_dw = np.dot(dL_dz, a_prev.T)
            dL_db = dL_dz

            layer.layer_weight_matrix -= (self.learning_rate*dL_dw)
            layer.layer_biases -= self.learning_rate*dL_db

            dL_da = np.dot(layer.layer_weight_matrix.T, dL_dz)
            
    # Batch size = 1, Stochastic G.D. for simplicity
    # note: epoch = training the NN on all batches once
    def train(self, initial_input: np.ndarray, target_output: np.ndarray, epochs: int):
        
        if initial_input.ndim == 1:
            initial_input = initial_input.reshape(-1, 1)
        if target_output.ndim == 1:
            target_output = target_output.reshape(-1, 1)

        for epoch in range(epochs):
            total_loss = 0  # Accumulate loss for this epoch

            for training_example, target_example in zip(initial_input, target_output):
                training_example = training_example.reshape(initial_input.shape[1], 1)
                target_example = target_example.reshape(target_output.shape[1], 1)

                prediction = self.forward_prop(training_example)
                self.backward_prop(training_example, prediction, target_example)

                training_sample_loss = loss_functions[self.loss](prediction, target_example, deriv=False)
                total_loss += training_sample_loss

            average_loss = total_loss / len(initial_input)

            if epoch % (epochs // 10) == 0:
                print(f"Epoch {epoch}/{epochs} | Avg Loss: {average_loss:.4f} ...")


    def predict(self, X: np.ndarray):
        # X always need to be a col vector of (input_shape, 1)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        return self.forward_prop(X)