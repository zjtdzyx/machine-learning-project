# 题目说明
# 泰坦尼克号生存预测（XGBoost）
# 本项目使用 XGBoost（梯度提升机） 来对 泰坦尼克号数据集 进行分类预测，目的是预测乘客是否幸存。你将与 随机森林 模型进行比较，分析并评估 XGBoost 模型在该任务中的表现，并计算多个评估指标。

# 导入必要的库
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 评估模型性能
def evaluate_model(y_test, y_pred):
    """
    评估模型的性能并输出相关指标。
    
    参数:
    y_test: 实际标签
    y_pred: 预测标签
    
    输出:
    打印准确率、精度、召回率、F1分数、ROC AUC等指标。
    """
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

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

# 评估XGBoost模型
evaluate_model(y_test, y_pred)
