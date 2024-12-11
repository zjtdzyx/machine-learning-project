# 练习题目：线性回归 - 波士顿房价预测 - 模型评估
# 评估线性回归模型的误差和准确性，并分析模型残差

# 导入所需的库
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

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
mae = mean_absolute_error(y_test, y_pred)  # 平均绝对误差
r2 = r2_score(y_test, y_pred)  # R²评分

# 打印评估指标
print("模型评估指标：")
print(f"均方误差 (MSE): {mse}")
print(f"均方根误差 (RMSE): {rmse}")
print(f"平均绝对误差 (MAE): {mae}")
print(f"R²: {r2}")

# 误差分析：绘制残差图
plt.figure(figsize=(10, 6))

# 残差计算（预测值与实际值之间的差异）
residuals = y_test - y_pred

# 绘制残差图
plt.scatter(y_pred, residuals, color='blue', alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--')  # y=0的参考线
plt.title("残差图：预测值与实际值之间的差异")
plt.xlabel("预测房价")
plt.ylabel("残差（实际值 - 预测值）")
plt.show()

# 可视化预测结果 vs 实际结果
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')  # 理想预测线
plt.title("实际房价与预测房价的对比")
plt.xlabel("实际房价")
plt.ylabel("预测房价")
plt.show()

