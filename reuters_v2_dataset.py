import numpy as np
from keras.datasets import reuters
from keras import models, layers


def vectorize(alist, dim=10000):
    a = np.zeros((len(alist), dim))
    for k, v in enumerate(alist):
        a[k, v] = 1
    return a


(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

x_train = vectorize(train_data)
x_test = vectorize(test_data)

y_train = np.array(train_labels)
y_test = np.array(test_labels)

x_val = x_train[:1000]
partial_x_val = x_train[1000:]
y_val = y_train[:1000]
partial_y_val = y_train[1000:]


model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(46, activation='softmax'))

model.compile(
    optimizer='rmsprop',
    loss='sparse_categorical_crossentropy',
    metrics=['acc']
)

model.fit(
    partial_x_val,
    partial_y_val,
    epochs=9,
    batch_size=512,
    validation_data=(x_val, y_val)
)

results = model.evaluate(x_test, y_test)
print(results)
