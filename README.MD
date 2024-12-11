# 机器学习算法与实践

本项目包含了六种常见的机器学习算法，用于分类任务的实现与评估。每种算法的实现都包括代码、模型训练、评估及相关的可视化过程，涵盖了以下内容：

1. **线性回归**
2. **决策树**
3. **随机森林**
4. **梯度提升（Gradient Boosting）**
5. **神经网络（MLP）**
6. **卷积神经网络（CNN）**

每种算法都在不同的数据集上进行训练和评估：
- **Iris数据集**：用于线性回归、决策树、随机森林、梯度提升。
- **Titanic数据集**：用于随机森林。
- **MNIST数据集**：用于神经网络（MLP）。
- **CIFAR-10数据集**：用于卷积神经网络（CNN）。

## 项目结构

```
ML_ALGORITHMS_AND_PRACTICES/
│
├── src/
│   ├── 1_linear_regression/
│   │   ├── linear_regression_code.py         # 线性回归模型实现
│   │   ├── linear_regression_evaluation.py   # 线性回归模型评估
│   │   └── linear_regression_model.py        # 线性回归模型保存与加载
│   ├── 2_decision_tree/
│   │   ├── decision_tree_code.py             # 决策树模型实现
│   │   ├── decision_tree_evaluation.py       # 决策树模型评估
│   │   └── decision_tree_model.py            # 决策树模型保存与加载
│   ├── 3_random_forest/
│   │   ├── random_forest_code.py             # 随机森林模型实现
│   │   ├── random_forest_evaluation.py       # 随机森林模型评估
│   │   └── random_forest_model.py            # 随机森林模型保存与加载
│   ├── 4_gradient_boosting/
│   │   ├── gradient_boosting_code.py         # 梯度提升模型实现
│   │   ├── gradient_boosting_evaluation.py   # 梯度提升模型评估
│   │   └── gradient_boosting_model.py        # 梯度提升模型保存与加载
│   ├── 5_neural_network/
│   │   ├── neural_network_code.py            # 神经网络（MLP）实现
│   │   ├── neural_network_evaluation.py      # 神经网络（MLP）评估
│   │   └── neural_network_model.py           # 神经网络（MLP）模型保存与加载
│   └── 6_cnn/
│       ├── cnn_code.py                       # 卷积神经网络（CNN）实现
│       ├── cnn_evaluation.py                 # 卷积神经网络（CNN）评估
│       └── cnn_model.py                      # 卷积神经网络（CNN）模型保存与加载
│
├── environment.yml                          # Conda环境配置文件
├── README.md                                # 项目文档
└── requirements.txt                         # Python依赖库
```

## 安装

你可以通过以下两种方式之一安装并配置本项目所需的环境。

### 方法 1：使用 Conda（推荐）

通过 `environment.yml` 文件创建一个新的 Conda 环境：

```bash
conda env create -f environment.yml
conda activate machine-learning-env
```

### 方法 2：使用 pip

你也可以通过 `requirements.txt` 文件使用 `pip` 安装所有依赖项：

```bash
pip install -r requirements.txt
```

请确保已安装 Python 3.8 及以上版本。

## 使用

### 1. 线性回归（Linear Regression）

`linear_regression_code.py` 脚本展示了如何使用线性回归模型进行训练，并对模型进行评估。运行脚本后，会输出模型的评估指标，比如 R 方值。

```bash
python src/1_linear_regression/linear_regression_code.py
```

### 2. 决策树（Decision Tree）

`decision_tree_code.py` 脚本使用 Iris 数据集训练一个决策树模型，并输出模型的准确率和混淆矩阵等评估结果。

```bash
python src/2_decision_tree/decision_tree_code.py
```

### 3. 随机森林（Random Forest）

`random_forest_code.py` 脚本使用 Titanic 数据集训练一个随机森林模型，评估模型的准确率、精度、召回率，并展示混淆矩阵。

```bash
python src/3_random_forest/random_forest_code.py
```

### 4. 梯度提升（Gradient Boosting）

`gradient_boosting_code.py` 脚本使用 Iris 数据集训练一个梯度提升（XGBoost）模型，评估模型的准确率，并展示特征重要性。

```bash
python src/4_gradient_boosting/gradient_boosting_code.py
```

### 5. 神经网络（Neural Network）

`neural_network_code.py` 脚本展示了如何使用神经网络（MLP）对 MNIST 数据集进行数字分类训练，并可视化训练过程中的准确率与损失曲线。

```bash
python src/5_neural_network/neural_network_code.py
```

### 6. 卷积神经网络（CNN）

`cnn_code.py` 脚本展示了如何使用卷积神经网络（CNN）对 CIFAR-10 数据集进行图像分类，训练过程中会输出训练与验证集的损失与准确度曲线。

```bash
python src/6_cnn/cnn_code.py
```

## 模型保存与加载

### 保存模型

每个算法的训练模型都可以保存为文件：

- **线性回归、决策树、随机森林、梯度提升（XGBoost）模型** 使用 `joblib` 保存：

  ```python
  import joblib
  joblib.dump(model, 'model.joblib')  # 保存模型
  ```

- **神经网络（MLP）和卷积神经网络（CNN）** 使用 Keras 的 `.h5` 格式保存：

  ```python
  model.save('model.h5')  # 保存 Keras 模型
  ```

### 加载模型

你可以通过以下方式加载已保存的模型：

- **对于 scikit-learn 模型（线性回归、决策树、随机森林、梯度提升）**：

  ```python
  import joblib
  model = joblib.load('model.joblib')  # 加载模型
  ```

- **对于 Keras 模型（神经网络、卷积神经网络）**：

  ```python
  from tensorflow.keras.models import load_model
  model = load_model('model.h5')  # 加载 Keras 模型
  ```

### 使用保存的模型进行预测

一旦模型被保存并加载，你可以使用加载的模型进行预测。假设你已经加载了一个训练好的模型，可以使用以下代码进行预测：

#### 示例：对新数据进行预测

```python
# 假设 `model` 已经被加载并准备好进行预测

# 例如，预测一张图片或一组数据
prediction = model.predict(new_data)  # new_data 需要与训练时的数据形状相同

# 如果是分类任务，获取预测的类别
predicted_class = tf.argmax(prediction, axis=-1).numpy()

print(f"Predicted class: {predicted_class}")
```

对于回归任务，直接使用模型的预测值：

```python
predicted_value = model.predict(new_data)
print(f"Predicted value: {predicted_value}")
```

## 依赖

### Python 库

本项目依赖于以下 Python 库：

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `xgboost`
- `tensorflow`
- `joblib`
- `scipy`

这些依赖项已在 `requirements.txt` 和 `environment.yml` 文件中列出。

### 环境设置

你可以使用以下方法之一来设置项目的依赖环境：

- **使用 Conda（推荐）**：通过 `environment.yml` 文件创建 Conda 环境。
- **使用 pip**：直接通过 `requirements.txt` 安装所有依赖库。

## 评估指标

每个模型的评估包括以下指标：

- **准确率**：分类准确率，正确分类的样本比例。
- **混淆矩阵**：展示模型预测结果与真实标签的比较。
- **精度、召回率、F1 值**（分类任务）。
- **ROC 曲线与 AUC 值**（二分类任务）。

## 许可证

本项目使用 MIT 许可证，具体内容请参考 [LICENSE](LICENSE) 文件。

