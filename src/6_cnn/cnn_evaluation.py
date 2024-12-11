# 练习题目说明：
# 练习题目名称：CIFAR-10图像分类（卷积神经网络）
# 任务说明：使用卷积神经网络（CNN）对CIFAR-10数据集进行图像分类，评估模型准确率，并绘制训练过程中的损失和准确度曲线。

# 导入必要的库
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt

# 1. 加载CIFAR-10数据集
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# 2. 数据归一化处理
train_images, test_images = train_images / 255.0, test_images / 255.0

# 3. 加载之前训练好的模型（假设已经在第一部分训练）
model = tf.keras.models.load_model("cifar10_cnn_model.h5")

# 4. 评估模型性能
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

# 输出模型在测试集上的准确率和损失
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# 5. 可视化预测结果
# 选择10个测试图像进行展示
plt.figure(figsize=(10, 5))
for i in range(10):
    ax = plt.subplot(2, 5, i + 1)
    ax.imshow(test_images[i])
    ax.set_title(f"True: {test_labels[i][0]}")
    ax.axis("off")
plt.show()

# 6. 打印一些预测的类别
predictions = model.predict(test_images[:10])  # 预测前10张图片
predicted_labels = [tf.argmax(pred, axis=-1).numpy() for pred in predictions]

print("Predicted labels for first 10 images:", predicted_labels)
