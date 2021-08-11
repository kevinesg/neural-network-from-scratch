from nn import NeuralNetwork
import numpy as np

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])

nn = NeuralNetwork([2, 2, 1], lr=0.5)
nn.fit(X, y, batch_size=2, epochs=20000)

for (x, label) in zip(X, y):
    pred = nn.predict(x)[0][0]
    step = 1 if pred > 0.5 else 0
    print(f'[INFO] data={x}, ground truth={label[0]}, pred={pred:.4f}, step={step}')