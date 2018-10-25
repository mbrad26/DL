import time
import numpy as np
import matplotlib.pyplot as plt
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


model = build_model()
model.fit(train_data,
          train_targets,
          epochs=500,
          batch_size=16
          )
mse_val, mae_val = model.evaluate(test_data, test_targets)
print(mse_val, mae_val)


#
# k = 4
# num_samples = len(train_data) // k
# num_epochs = 500
# all_mae_history = []
#
# for i in range(k):
#     print('Processing fold #{}'.format(i))
#     val_data = train_data[i*num_samples: (i+1)*num_samples]
#     val_targets = train_targets[i*num_samples: (i+1)*num_samples]
#     partial_train_data = np.concatenate(
#         [train_data[:i*num_samples], train_data[(i+1)*num_samples:]], axis=0
#     )
#     partial_train_targets = np.concatenate([train_targets[:i*num_samples],
#                                             train_targets[(i+1)*num_samples:]], axis=0)
#     model = build_model()
#     history = model.fit(
#         partial_train_data,
#         partial_train_targets,
#         epochs=num_epochs,
#         batch_size=1,
#         validation_data=(val_data, val_targets),
#         verbose=0
#     )
#     mae_history = history.history['val_mean_absolute_error']
#     all_mae_history.append(mae_history)
#
# avg_mae_history = [np.mean([x[i] for x in all_mae_history]) for i in range(num_epochs)]
# q
#
# plt.plot(range(1, len(avg_mae_history) + 1), avg_mae_history)
# plt.xlabel('Epochs')
# plt.ylabel('Validation MAE')
# plt.show()
#
#
# def smooth_curve(points, factor=0.9):
#     smoothed_points = []
#     for point in points:
#         if smoothed_points:
#             previous = smoothed_points[-1]
#             smoothed_points.append(previous * factor + point * (1 - factor))
#         else:
#             smoothed_points.append(point)
#     return smoothed_points
#
#
# smooth_mae_history = smooth_curve(avg_mae_history[10:])
#
# plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
# plt.xlabel('Epochs')
# plt.ylabel('Validation MAE')
# plt.show()


end = time.time()
print('Duration:', (end - start) / 60, ' minutes')



































