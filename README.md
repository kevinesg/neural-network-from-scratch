# Neural Network from scratch
The neural network can be found at `nn.py`. The tunable hyperparameters are number of layers, number of nodes of each layer, learning rate, batch size, and choice between `mean squared error` and `binary crossentropy` as the loss function. The activation function of all layers is sigmoid. I plan on adding more loss functions and activation functions as well.

The neural network can be used to train and predict XOR values. `predict_xor.py` can be used to check this. `predict_mini_MNIST.py` trains a "mini" version of the MNIST dataset (8x8 pixels and less sample size). `predict_MNIST.py` on the other hand uses the whole training set from Kaggle. Both datasets can achieve 99% validation set accuracy in less than 100 epochs.

The neural network architecture is heavily inspired by the book `Deep Learning for Computer Vision` by Adrian Rosebrock. In fact, majority of the code came from there but I changed some lines and added batch training and categorical crossentropy loss, as compared to the original architecture which trains the dataset one sample at a time and the only loss function is sum of squared error.
#
If you have any questions or suggestions, feel free to contact me here. Thanks for reading!
