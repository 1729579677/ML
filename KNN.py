import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

# 读取xlsx文件
data = pd.read_excel("data.xlsx")


data = data[data['label1'] != 0]
data = data[data['label1'] != 4]

# 合并标题和摘要
data['text'] = data['title'] + ' ' + data['abstract']

# 填充空值
data['label2'].fillna('', inplace=True)
data['label3'].fillna('', inplace=True)

# 将标签编码为数字
label_encoder = LabelEncoder()
data['label1'] = label_encoder.fit_transform(data['label1'])
#data['label2'] = label_encoder.transform(data['label2'])
#data['label3'] = label_encoder.transform(data['label3'])


# 设置显示选项以显示所有列
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print(data)



# 划分特征和标签
X = data['text']
y = data['label1']

# TF-IDF向量化
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.1, random_state=42)

# 初始化并训练KNN分类器
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)

# 进行预测
y_pred = knn_classifier.predict(X_test)

# 输出分类报告和准确率
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
