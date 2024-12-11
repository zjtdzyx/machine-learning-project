# 评估模型的准确率与混淆矩阵

# 导入必要的库
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# 1. 加载数据集
iris = load_iris()
X = iris.data  # 特征数据：花萼和花瓣的长度与宽度
y = iris.target  # 标签数据：鸢尾花的种类

# 2. 数据拆分：80%训练，20%测试
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 训练决策树模型
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# 4. 可视化决策树
plt.figure(figsize=(15, 10))
plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.title("Decision Tree Visualization")
plt.show()

# 使用测试集进行预测
y_pred = clf.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"模型的准确率：{accuracy:.2f}")

# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# 计算分类报告：包括精度、召回率和F1分数
from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred, target_names=iris.target_names)
print("分类报告：")
print(report)

