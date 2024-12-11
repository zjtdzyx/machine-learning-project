# 练习题目说明：
# 题目名称：MNIST手写数字识别（神经网络）
# 使用训练好的神经网络模型，在测试集上进行评估，输出模型的损失值和准确率。

# 导入必要的库
import tensorflow as tf
from tensorflow.keras.models import load_model

# 加载MNIST数据集
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 数据预处理：将图像像素值缩放到[0, 1]之间
x_test = x_test / 255.0

# 假设我们之前已经训练并保存了模型，现在加载它
# model = load_model('path_to_saved_model')  # 若有保存模型的需求可以使用此行代码

# 直接使用之前训练的模型
# 在测试集上评估模型
test_loss, test_accuracy = model.evaluate(x_test, y_test)

# 输出测试集上的评估结果
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

# 预测部分：使用模型进行预测，输出前10个预测结果
predictions = model.predict(x_test[:10])

# 显示前10个预测结果与实际标签
print("\nPredicted labels for first 10 images:")
for i in range(10):
    print(f"Image {i+1}: Predicted: {tf.argmax(predictions[i]).numpy()}, Actual: {y_test[i]}")
