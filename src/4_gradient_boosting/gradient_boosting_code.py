# 题目说明
# 泰坦尼克号生存预测（XGBoost）
# 本项目使用 XGBoost（梯度提升机） 来对 泰坦尼克号数据集 进行分类预测，目的是预测乘客是否幸存。你将与 随机森林 模型进行比较，分析并评估 XGBoost 模型在该任务中的表现，并计算多个评估指标。

# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# 数据集URL
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

# 读取数据集
train_df = pd.read_csv(url)

# 数据预处理
# 填补缺失值
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)
train_df['Fare'].fillna(train_df['Fare'].median(), inplace=True)

# 选择特征列
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = train_df[features]

# 将分类变量转换为数值类型
X['Sex'] = LabelEncoder().fit_transform(X['Sex'])
X['Embarked'] = LabelEncoder().fit_transform(X['Embarked'])

# 目标变量
y = train_df['Survived']

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化XGBoost分类器
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

# 训练模型
xgb_model.fit(X_train, y_train)

# 进行预测
y_pred = xgb_model.predict(X_test)

# 计算评估指标
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

# 输出评估指标
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'ROC AUC: {roc_auc:.4f}')

# 绘制混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=["Not Survived", "Survived"], yticklabels=["Not Survived", "Survived"])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# 可视化特征重要性
xgb.plot_importance(xgb_model, importance_type='weight', title="Feature Importance", height=0.8)
plt.show()
