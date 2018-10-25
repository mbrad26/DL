import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import imdb
from keras import models, layers, regularizers

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


def vectorize(alist, dim=10000):
    a = np.zeros((len(alist), dim))
    for k, v in enumerate(alist):
        a[k, v] = 1
    return a


x_train = vectorize(train_data)
x_test = vectorize(test_data)
y_train = np.array(train_labels).astype('float32')
y_test = np.array(test_labels).astype('float32')

x_val = x_train[:10000]
partial_train_data = x_train[10000:]
y_val = y_train[:10000]
partial_train_labels = y_train[10000:]


model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=7, batch_size=512, validation_data=(x_test, y_test))

history_dict = history.history
loss = history_dict['loss']
acc = history_dict['acc']
val_loss = history_dict['val_loss']
val_acc = history_dict['val_acc']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('imdb_v1_loss_16')
plt.show()

plt.plot(epochs, acc, 'bo', label='Training Acc')
plt.plot(epochs, val_acc, 'b', label='Validation Acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('imdb_v1_acc_16')
plt.show()