# 练习题目说明：
# 练习题目名称：CIFAR-10图像分类（卷积神经网络）
# 任务说明：使用卷积神经网络（CNN）对CIFAR-10数据集进行图像分类，评估模型准确率，并绘制训练过程中的损失和准确度曲线。

# 导入必要的库
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

# 5. 可视化训练过程中的损失和准确度曲线
# 绘制训练和验证集上的损失曲线
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 绘制训练和验证集上的准确度曲线
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# 6. 评估模型
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

# 打印评估结果
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")
