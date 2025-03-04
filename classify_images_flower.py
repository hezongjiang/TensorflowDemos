import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib

# 常量
batch_size = 32
img_height = 180
img_width = 180
epochs = 10


# 下载并查看数据集
def download_image():
    # 包含约 3,700 张花卉照片的数据集。该数据集包含 5 个子目录，每个子目录对应一个类
    # flower_photo/
    #   daisy/
    #   dandelion/
    #   roses/
    #   sunflowers/
    #   tulips/
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file('flower_photos.tar', origin=dataset_url, extract=True)
    data_dir = pathlib.Path(data_dir).with_suffix('')
    # 下载后，应该拥有一个数据集的副本。总共有 3,670 个图像：
    image_count = len(list(data_dir.glob('*/*.jpg')))
    print(image_count)
    # 下面是一些玫瑰：
    roses = list(data_dir.glob('roses/*'))
    PIL.Image.open(str(roses[0]))
    PIL.Image.open(str(roses[1]))
    # 一些郁金香
    tulips = list(data_dir.glob('tulips/*'))
    PIL.Image.open(str(tulips[0]))
    PIL.Image.open(str(tulips[1]))
    return data_dir


# 切分数据集
def split_train_test(data_dir):
    # 将 80% 的图像用于训练，将 20% 的图像用于验证。
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    return train_ds, val_ds


# 训练模型
def fit():
    # 编译模型
    # 选择 tf.keras.optimizers.Adam 优化器和 tf.keras.losses.SparseCategoricalCrossentropy 损失函数。要查看每个训练周期的训练和验证准确率，请将 metrics 参数传递给 Model.compile。
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.summary()
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    return history


def plt_image(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(epochs)
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


# 下载数据
data_dir = download_image()
# 分割训练集、测试集
train_ds, val_ds = split_train_test(data_dir)

class_names = train_ds.class_names
print(class_names)

# 下面是训练数据集中的 9 个图像
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

# image_batch 是形状为 (32, 180, 180, 3) 的张量。这是由 32 个形状为 180x180x3（最后一个维度是指颜色通道 RGB）的图像组成的批次。
# label_batch 是形状为 (32,) 的张量，这些是 32 个图像的对应标签。
for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

num_classes = len(class_names)
# 基本 Keras 模型
# 创建模型
# Keras 序贯模型由三个卷积块 (tf.keras.layers.Conv2D) 组成，每个卷积块都有一个最大池化层 (tf.keras.layers.MaxPooling2D)。
# 有一个全连接层 (tf.keras.layers.Dense)，上方有 128 个单元，由 ReLU 激活函数 ('relu') 激活。
model = Sequential([
    layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

AUTOTUNE = tf.data.AUTOTUNE
# 缓存并打乱数据顺序可以避免模型学习到数据的顺序特征，从而提高模型的泛化能力。
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 训练
history = fit()
# 在训练集和验证集上创建损失和准确率的图表
plt_image(history)

# 数据增强
# 过拟合通常会在训练样本数量较少的情况下发生。数据增强采用的方法是：通过增强然后使用随机转换，从现有样本中生成其他训练数据，产生看起来可信的图像。这有助于向模型公开数据的更多方面，且有助于更好地进行泛化。
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ]
)

# 通过对同一图像多次应用数据增强来呈现一些增强示例
plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")

# 重新定义模型
model = Sequential([
    data_augmentation,
    layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

# 重新训练
history = fit()
# 画图
plt_image(history)

# save model
model_name = 'image_classify_cnn.model'
model.save(model_name)


# load model
# new_model = tf.keras.models.load_model(model_name)

# 预测
def predict_image(flower_url):
    sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=flower_url)
    img = tf.keras.utils.load_img(sunflower_path, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    return score


score = predict_image("https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg")

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
