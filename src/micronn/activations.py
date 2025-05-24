import numpy as np


def sigmoid(x: np.ndarray, deriv: bool = False) -> np.ndarray:
    if not deriv:
        return 1/(1 + np.exp(-x))
    else:
        return sigmoid(x)*(1-sigmoid(x))

def relu(x: np.ndarray, deriv: bool = False) -> np.ndarray:
    if not deriv:
        return np.maximum(0, x)
    else:
        # Derivative of ReLU: 1 if x > 0, 0 otherwise
        return np.where(x > 0, 1.0, 0.0)

def tanh(x: np.ndarray, deriv: bool = False) -> np.ndarray:
    t = np.tanh(x)
    if not deriv:
        return t
    else:
        return 1 - t**2


# Yeah, I wasn't gonna write this on my own

##############################################################################################################

def softmax(x: np.ndarray, deriv: bool = False) -> np.ndarray:
    # Shift x to avoid overflow/underflow before exponentiation
    # Subtracting max(x) makes the largest element 0, so exp(0)=1
    # This is a standard numerical stability trick for softmax.
    shifted_x = x - np.max(x, axis=-1, keepdims=True)
    e_x = np.exp(shifted_x)
    s = e_x / np.sum(e_x, axis=-1, keepdims=True)
    
    if not deriv:
        return s
    else:
        # NOTE: The derivative of softmax is a bit more complex.
        # If softmax is used as the output layer with Categorical Cross-Entropy loss,
        # the derivative of the loss with respect to the pre-activation (z)
        # simplifies to (softmax_output - true_labels_one_hot).
        #
        # The derivative da_i/dz_j (where a is softmax output, z is input to softmax)
        # is s_i * (delta_ij - s_j), where delta_ij is the Kronecker delta.
        # This forms a Jacobian matrix.
        #
        # For an element-wise multiplication as in your current backprop:
        # dL_dz = dL_da * da_dz
        # A common simplification (though not fully accurate for the off-diagonal elements
        # of the Jacobian) is to use s * (1 - s), similar to sigmoid.
        # This represents the diagonal elements of the Jacobian.
        # This simplification is often used in pedagogical examples or when
        # the non-diagonal interactions are assumed to be less critical or are
        # handled implicitly by the overall optimization.
        #
        # If you use Categorical Cross-Entropy, you might want to adjust your
        # backpropagation for the last layer to directly compute dL/dz.
        return s * (1 - s)

##############################################################################################################