import numpy as np
import warnings
from typing import List
from activations import sigmoid, relu, tanh, softmax


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
                self.layer[i].bias = 0

    # Check the weights and biases of each Neuron in Dense layer
    def spit_weights(self):
        for i in range(self.units):
            print(f"For {i+1}th Neuron: w = {self.layer[i].weights}, b = {self.layer[i].bias}")

    def forward(self, input: np.ndarray) -> np.ndarray:
        output = np.dot(self.layer_weight_matrix, input) + np.array([self.layer[i].bias for i in range(self.units)]).reshape(self.units, 1)
        
        if self.activation == "sigmoid":
            return sigmoid(output)
        
        elif self.activation == "relu":
            return relu(output)
        
        elif self.activation == "tanh":
            return tanh(output)
        
        elif self.activation == "softmax":
            return softmax(output)
        
        else:
            raise ValueError("Only sigmoid, tanh, softmax and ReLU activation functions are supported for now :(")




# FLAW: the input_shape param will be defined only for the first Dense layer in the Model(List[Dense])
class Model():
    layer_pointer = 0

    def __init__(self, layers: List[Dense], loss: str):
        
        if not layers:
            raise ValueError("Sequential model must contain at least one layer.")

        # Ensure the first layer has an input_shape defined
        if not hasattr(layers[0], 'input_shape') or layers[0].input_shape is None:
            raise ValueError("First Dense layer must define an 'input_shape'.")
        
        warnings.warn("INPUT_SHAPE PARAMETER VALUE PROVIDED FOR ALL DENSE LAYERS EXCEPT THE FIRST WILL BE OVERIDDEN!")

        self.layers = layers
        self.loss = loss
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
        
    ### Backward_Prop
    ### loss

    # Batch size = 1, Stochastic G.D. for simplicity
    # note: epoch = training the NN on all batches once
    def train(self, initial_input: np.ndarray, epochs: int, learning_rate: float):
        pass

# dense_layer_1 = Dense(units=32, activation="sigmoid", input_shape=2)
# dense_layer_1.spit_weights()

model = Model(
    [
    Dense(units=10, activation="sigmoid", input_shape=5),
    Dense(units=2, activation="softmax")], 
    loss="placeholder"
    )

for layer in model.layers:
    layer.spit_weights()