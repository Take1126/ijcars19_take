# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#参考:https://github.com/tensorflow/tensorflow/issues/34888
import tensorflow as tf
#UnknownError回避用
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)

import numpy as np
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
from tensorflow import keras
fashion_mnist = keras.datasets.fashion_mnist
#ダウンロード終了後からの実行時間計測のために追加します。
import time
start_time = time.time()

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0
train_images = np.expand_dims(train_images, axis=3)
test_images = np.expand_dims(test_images, axis=3)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28,28,1)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.summary()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, batch_size=64, epochs=10, validation_data=(test_images, test_labels))

# 計測した結果を出力
tat_time = time.time() - start_time
print ("実行時間:{0}".format(tat_time) + "[秒]")