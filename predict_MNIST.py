from nn import NeuralNetwork
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

data_df = pd.read_csv('train.csv')
data = np.array(data_df)
y = data[:, 0]
y = y.reshape(y.shape[0], 1)
X = data[:, 1:] / 255
y = LabelBinarizer().fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

print('[INFO] training network...')
nn = NeuralNetwork([X_train.shape[1], 512, 100, 10])
print(f'[INFO] {nn}')
nn.fit(X_train, y_train, batch_size=32, epochs=100, verbose=1)

print('[INFO] evaluating network...')
preds = nn.predict(X_test)
preds = preds.argmax(axis=1)
print(classification_report(y_test.argmax(axis=1), preds))