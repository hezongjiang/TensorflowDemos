import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# load data
datapath = '/data/mnist/mnist.npz'
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data(datapath)

# show first image
plt.imshow(train_images[0])
plt.show()

# normalize 0~1
train_images = tf.keras.utils.normalize(train_images, axis=1)
test_images = tf.keras.utils.normalize(test_images, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# training
model.fit(train_images, train_labels, epochs=3)

val_loss, val_acc = model.evaluate(test_images, test_labels)
print(val_loss)
print(val_acc)

# save model
model.save('epic_num_reader.model')

# load model
new_model = tf.keras.models.load_model('epic_num_reader.model')

# show some one
i = 1000
plt.imshow(test_images[i], cmap=plt.cm.binary)
plt.show()

# predict
predictions = new_model.predict(test_images)
# predictions = model.predict(test_images)
print(np.argmax(predictions[i]))
