import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import reuters
from keras import models, layers
from keras.utils import to_categorical


(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)


def vectorize(alist, dim=10000):
    a = np.zeros((len(alist), dim))
    for k, v in enumerate(alist):
        a[k,v] = 1
    return a


def to_one_hot(alist, dim=46):
    b = np.zeros((len(alist), dim))
    for k, v in enumerate(alist):
        b[k, v] = 1
    return b


x_train = vectorize(train_data)
x_test = vectorize(test_data)

# y_train = np.array(train_labels)
# y_test = np.array(test_labels)

one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)

# one_hot_train_labels = to_categorical(train_labels)
# one_hot_test_labels = to_categorical(test_labels)

x_val = x_train[:1000]
partial_x_val = x_train[1000:]
y_val = one_hot_train_labels[:1000]
partial_y_val = one_hot_train_labels[1000:]


model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['acc']
)


results = model.evaluate(x_test, one_hot_test_labels)
print(results)

history = model.fit(
    partial_x_val,
    partial_y_val,
    epochs=20,
    batch_size=512,
    validation_data=(x_val, y_val)
)

history_dict = history.history
loss = history_dict['loss']
acc = history_dict['acc']
val_loss = history_dict['val_loss']
val_acc = history_dict['val_acc']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


plt.plot(epochs, acc, 'bo', label='Training Acc')
plt.plot(epochs, val_acc, 'b', label='Validation Acc')
plt.title('Training and validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()
plt.show()
