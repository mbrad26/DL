import numpy as np
import time
from keras.datasets import boston_housing
from keras import models, layers

start = time.time()

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std


def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))

    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


k = 4
num_samples = len(train_data) // k
num_epochs = 100
all_scores = []


for i in range(k):
    print('Processing fold #{}'.format(i))
    val_data = train_data[i * num_samples: (i + 1) * num_samples]
    val_targets = train_targets[i * num_samples: (i + 1) * num_samples]
    partial_train_data = np.concatenate([train_data[:i * num_samples],
                                         train_data[(i + 1) * num_samples:]], axis=0)
    partial_train_targets = np.concatenate([train_targets[:i * num_samples],
                                            train_targets[(i + 1) * num_samples:]], axis=0)
    model = build_model()
    model.fit(partial_train_data,
              partial_train_targets,
              epochs=num_epochs,
              batch_size=1,
              verbose=1)
    val_mse, val_mae = model.evaluate(val_data, val_targets)
    all_scores.append(val_mae)

end = time.time()
print(end - start)