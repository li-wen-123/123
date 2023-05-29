import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 读取数据集
data = pd.read_csv("your_dataset.csv")  # 替换为您的数据集路径或名称

# 提取特征和标签
X = data.drop("phase_labels", axis=1)  # 特征
y = data["phase_labels"]  # 标签

# 数据预处理
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建神经网络模型
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(64, activation="relu", input_dim=X_train.shape[1]))
model.add(tf.keras.layers.Dense(32, activation="relu"))
model.add(tf.keras.layers.Dense(y_train.shape[1], activation="sigmoid"))

# 编译模型
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# 训练模型
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# 在测试集上进行预测
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)  # 将概率转换为二进制标签

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("准确率：", accuracy)