# microNN
a barebones implementation for training feedforward NNs from scratch.

Built in a weekend to understand the flow (and pain) of building a mini-NN framework.

<h3>> 
 ⚠️ This is a quick built-in-a-weekend implementation — so don't judge too much ;) 
 
 - Only supports sigmoid, ReLU, tanh, softmax(untested) activations, MSE loss
 
 - manual differentiation and batch size = 1 for now.

 -  Trains XOR in 100k epochs in little under 1min.
</h3>

## Download it!

- `git clone https://github.com/freebirdyeah/microNN/`
- `cd microNN/`
- `pip install -e .`

<h4 >To uninstall:- </h4>

- `pip uninstall micronn`

## Features
- Dense layers
- Sigmoid, ReLU, tanh activation
- MSE loss
- Stochastic Gradient Descent (`batch_size` = 1)

## Near Goals
- more testing (especially with softmax)
- add mini-batch gradient descent
- automatic one-hot-encoding on arrays of boolean values
- add CategoricalCrossEntropy Loss function
 
## Future Goals
- Add an optimizer like `Adam`
- maybe think about adding an autograd engine in the future

## Example Code:-

```python
import numpy as np
from micronn import Model, Dense


X = np.array([
    [0, 0],
    [1, 1],
    [0, 1], 
    [1, 0]
])


Y = np.array([[0], 
              [0], 
              [1], 
              [1]])


model = Model(
    layers=[
        Dense(units=4, activation="sigmoid", input_shape=2),
        Dense(units=1, activation="sigmoid")
    ],
    loss="MSE",
    learning_rate=0.01
)

model.train(X, Y, epochs=100_000)

example_test = np.array([0, 1])

# rounding to 0/1
prediction = np.round(model.predict(example_test))
print(prediction)
```

<h3>Output:-</h3>

```
UserWarning: INPUT_SHAPE PARAMETER VALUE PROVIDED FOR ALL DENSE LAYERS EXCEPT THE FIRST WILL BE OVERIDDEN!
  warnings.warn("INPUT_SHAPE PARAMETER VALUE PROVIDED FOR ALL DENSE LAYERS EXCEPT THE FIRST WILL BE OVERIDDEN!")

Epoch 0/100000 | Avg Loss: 0.1333 ...
Epoch 10000/100000 | Avg Loss: 0.1185 ...
Epoch 20000/100000 | Avg Loss: 0.0833 ...
Epoch 30000/100000 | Avg Loss: 0.0291 ...
Epoch 40000/100000 | Avg Loss: 0.0084 ...
Epoch 50000/100000 | Avg Loss: 0.0039 ...
Epoch 60000/100000 | Avg Loss: 0.0024 ...
Epoch 70000/100000 | Avg Loss: 0.0016 ...
Epoch 80000/100000 | Avg Loss: 0.0012 ...
Epoch 90000/100000 | Avg Loss: 0.0010 ...
[[1.]]
```
