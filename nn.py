import numpy as np

class NeuralNetwork:
    # Initialize weight matrix, layers dimensions, and learning rate
    def __init__(self, layers, lr=0.1):
        self.W = []
        self.layers = layers    # List of number of nodes per layer
        self.lr = lr

        for i in range(0, len(layers) - 2):
            w = np.random.randn(layers[i] + 1, layers[i + 1] + 1)
            # Normalize the weights depending on the number of nodes per layer
            self.W.append(w / np.sqrt(layers[i]))
        
        # Last layer (no need to add bias)
        w = np.random.randn(layers[-2] + 1, layers[-1])
        # Normalize
        self.W.append(w / np.sqrt(layers[-1]))

    def __repr__(self):
        return f'NeuralNetwork: {"-".join(str(l) for l in self.layers)}'

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    # Input x should be the output of the sigmoid function
    def sigmoid_deriv(self, x):
        return x * (1 - x)
    
    def fit(self, X, y, batch_size=1, epochs=1000, verbose=100):
        # Add bias
        X = np.hstack([X, np.ones((X.shape[0], 1))])

        # Generate and randomize indexes for batch training
        indexes = np.arange(X.shape[0])
        for epoch in range(0, epochs):
            np.random.shuffle(indexes)
            for i in range(0, X.shape[0], batch_size):
                batch_indexes = indexes[i:i + batch_size]
                X_batch = X[batch_indexes]
                y_batch = y[batch_indexes]
                self.fit_partial(X_batch, y_batch)

            # Print updates about loss and accuracy
            if epoch == 0 or (epoch + 1) % verbose == 0:
                loss = self.calculate_loss(X, y, loss='binary_crossentropy')
                acc = self.calculate_accuracy(X, y)
                print(f'[INFO] epoch={epoch + 1}, loss={loss:.7f}, acc={acc:.4f}')
    
    def fit_partial(self, x, y):
        A = [np.atleast_2d(x)]

        # Feedforward
        for layer in range(0, len(self.W)):
            raw = A[layer].dot(self.W[layer])
            activated = self.sigmoid(raw)
            A.append(activated)
        
        # Compute gradients
        error = y - A[-1]
        D = [error * self.sigmoid_deriv(A[-1])]

        for layer in range(len(A) - 2, 0, -1):
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.sigmoid_deriv(A[layer])
            D.append(delta)

        # Reverse the gradients list since we constructed it backwards
        D = D[::-1]

        # Adjust weights
        for layer in range(0, len(self.W)):
            self.W[layer] += self.lr * A[layer].T.dot(D[layer])
    
    def predict(self, X, bias=True):
        p = np.atleast_2d(X)

        if bias == True:
            p = np.hstack([p, np.ones((p.shape[0], 1))])
        
        # Feedforward
        for layer in range(0, len(self.W)):
            p = self.sigmoid(p.dot(self.W[layer]))
        
        return p
    
    def calculate_loss(self, X, y, loss):
        y = np.atleast_2d(y)
        preds = self.predict(X, bias=False)

        if loss == 'mse':
            loss = ((preds - y) ** 2) / X.shape[0]
        elif loss == 'binary_crossentropy':
            loss = -1 / X.shape[0] * np.sum(y * np.log(preds) + (1 - y) * np.log(1 - preds))

        return loss

    def calculate_accuracy(self, X, y):
        y = np.atleast_2d(y)
        preds = self.predict(X, bias=False)
        preds = np.argmax(preds, axis=1)
        y = np.argmax(y, axis=1)
        acc = (preds == y).mean()
        
        return acc