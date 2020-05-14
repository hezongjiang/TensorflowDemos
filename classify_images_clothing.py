import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import gzip
import numpy as np
# import ssl
#
# ssl._create_default_https_context = ssl._create_unverified_context
#
# (train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

def load_data(data_folder):
    files = ['train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
             't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz']

    paths = []
    for f_name in files:
        paths.append(os.path.join(data_folder, f_name))

    with gzip.open(paths[0], 'rb') as lb_path:
        y_train = np.frombuffer(lb_path.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as img_path:
        x_train = np.frombuffer(img_path.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

    with gzip.open(paths[2], 'rb') as lb_path:
        y_test = np.frombuffer(lb_path.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as img_path:
        x_test = np.frombuffer(img_path.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)

    return (x_train, y_train), (x_test, y_test)

(train_images, train_labels), (test_images, test_labels) = load_data(os.path.dirname(__file__) + '/data/fashion')

# show first image
plt.figure()
# plt.imshow(train_images[0])
# plt.show()
# print(train_labels[0])

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# prepare data, scale these values to a range of 0 to 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# show first 25 image and label
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    # plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

predictions = model.predict(test_images)

idx = 20
plt.figure()
plt.imshow(train_images[idx])
plt.show()
print(predictions[idx])
predict = int(np.argmax(predictions[idx]))
print('predict: ' + class_names[predict])
print('label:   ' + class_names[test_labels[idx]])
