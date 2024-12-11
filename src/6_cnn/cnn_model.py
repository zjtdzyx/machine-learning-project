# 练习题目说明：保存卷积神经网络（CNN）模型
# 使用 TensorFlow 保存训练好的卷积神经网络模型（CIFAR-10分类）

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10

# 1. 加载和预处理数据
# 加载CIFAR-10数据集
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# 数据归一化处理，将像素值缩放到0到1之间
train_images, test_images = train_images / 255.0, test_images / 255.0

# 2. 创建卷积神经网络（CNN）模型
model = models.Sequential([
    # 第一层卷积层，使用32个3x3的卷积核
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    # 池化层，使用最大池化
    layers.MaxPooling2D((2, 2)),
    
    # 第二层卷积层，使用64个3x3的卷积核
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # 第三层卷积层，使用64个3x3的卷积核
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    # 展平层，将多维输入一维化
    layers.Flatten(),
    
    # 全连接层，节点数为64
    layers.Dense(64, activation='relu'),
    
    # 输出层，10个节点，分别对应CIFAR-10的10个类别
    layers.Dense(10)
])

# 3. 编译模型
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 4. 训练模型
history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# 5. 保存模型
model.save('cifar10_cnn_model.h5')

print("卷积神经网络模型已保存为 'cifar10_cnn_model.h5'")
