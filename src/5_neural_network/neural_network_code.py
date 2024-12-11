# 练习题目说明：
# 题目名称：MNIST手写数字识别（神经网络）
# 使用神经网络模型（MLP）对MNIST数据集进行手写数字识别，并可视化训练过程中的准确率和损失值。

# 导入必要的库
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# 加载MNIST数据集
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 数据预处理：将图像像素值缩放到[0, 1]之间
x_train, x_test = x_train / 255.0, x_test / 255.0

# 创建神经网络模型
model = Sequential([
    Flatten(input_shape=(28, 28)),  # 将28x28的图像数据展平为一维
    Dense(128, activation='relu'),  # 第一个隐藏层，128个神经元，激活函数使用ReLU
    Dense(10, activation='softmax') # 输出层，10个神经元，softmax用于多分类
])

# 编译模型
model.compile(optimizer=Adam(), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# 训练模型
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# 可视化训练过程中的损失值和准确率
plt.figure(figsize=(12, 5))

# 可视化训练和验证的准确率
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# 可视化训练和验证的损失值
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# 模型评估：在测试集上评估模型的表现
test_loss, test_accuracy = model.evaluate(x_test, y_test)

# 输出测试集上的评估结果
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')
