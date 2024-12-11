# 练习题目：线性回归 - 加利福尼亚房价预测 - 模型保存

# 导入所需的库
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing

# 加载加利福尼亚房价数据集
california_housing = fetch_california_housing()
X = california_housing.data  # 特征
y = california_housing.target  # 目标变量（房价）

# 将数据分为训练集和测试集（80%训练，20%测试）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建线性回归模型并进行训练
model = LinearRegression()
model.fit(X_train, y_train)

# 保存训练好的模型到文件
model_filename = 'linear_regression_california_model.pkl'
joblib.dump(model, model_filename)

print(f"模型已成功保存为 {model_filename}")
