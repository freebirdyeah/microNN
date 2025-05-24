import numpy as np


def MSE(prediction: np.ndarray, output: np.ndarray, deriv=False) -> np.ndarray:
    if not deriv:
        return 0.5*np.mean(np.square((prediction - output)))
    else:
        return prediction - output