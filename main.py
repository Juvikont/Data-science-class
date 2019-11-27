from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Flatten, Dense
from matplotlib import pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
x_train = x_train / 255.0 - 0.5
x_test = x_test / 255.0 - 0.5

model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(256, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile('adam', 'categorical_crossentropy', ['acc'])

train_acc = []
test_acc = []
for i in range(10):
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test))
    train_acc += history.history['acc']
    test_acc += history.history['val_acc']
    plt.plot(list(range(i + 1)), train_acc, list(range(i + 1)), test_acc)
    plt.show()
