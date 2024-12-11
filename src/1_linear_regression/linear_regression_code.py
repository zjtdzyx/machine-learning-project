# 练习题目：线性回归 - 波士顿房价预测
# 使用线性回归预测波士顿地区的房价，评估模型的R²得分和均方误差（MSE）

# 导入所需的库
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error, r2_score

# 加载波士顿房价数据集
boston = load_boston()
X = boston.data  # 特征
y = boston.target  # 目标变量（房价）

# 将数据分为训练集和测试集（80%训练，20%测试）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建线性回归模型并进行训练
model = LinearRegression()
model.fit(X_train, y_train)

# 使用训练好的模型进行预测
y_pred = model.predict(X_test)

# 计算模型的评估指标
mse = mean_squared_error(y_test, y_pred)  # 均方误差
rmse = np.sqrt(mse)  # 均方根误差
r2 = r2_score(y_test, y_pred)  # R²评分

# 打印评估指标
print(f"均方误差 (MSE): {mse}")
print(f"均方根误差 (RMSE): {rmse}")
print(f"R²: {r2}")

# 可视化结果：实际房价 vs 预测房价
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')  # 理想预测线
plt.title("实际房价与预测房价的对比")
plt.xlabel("实际房价")
plt.ylabel("预测房价")
plt.show()

