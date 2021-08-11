from nn import NeuralNetwork
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets

print('[INFO] loading MNIST (sample) dataset...')
# 8x8 pixels MNIST dataset images
digits = datasets.load_digits()
data = digits.data.astype('float')
data = (data - data.min()) / (data.max() - data.min())

labels = LabelBinarizer().fit_transform(digits.target)
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.25
)

print('[INFO] training network...')
nn = NeuralNetwork([X_train.shape[1], 64, 32, 10])
print(f'[INFO] {nn}')
nn.fit(X_train, y_train, batch_size=8, epochs=1000)

print('[INFO] evaluating network...')
preds = nn.predict(X_test)
preds = preds.argmax(axis=1)
print(classification_report(y_test.argmax(axis=1), preds))