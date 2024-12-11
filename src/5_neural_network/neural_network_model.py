# 练习题目说明：保存训练好的神经网络模型
# 使用 TensorFlow 将训练好的神经网络模型保存到一个文件夹中

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

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

# 保存模型
model.save('mnist_nn_model.h5')

print("神经网络模型已保存为 'mnist_nn_model.h5'")
