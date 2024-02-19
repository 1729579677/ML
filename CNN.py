import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dropout

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

# 划分特征和标签
X = data['text']
y = data['label1']

# 将文本转换为序列
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
X_seq = tokenizer.texts_to_sequences(X)

# 对序列进行填充
max_seq_length = max([len(seq) for seq in X_seq])
X_padded = pad_sequences(X_seq, maxlen=max_seq_length, padding='post')

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)

# 构建深度学习模型
embedding_dim = 100
vocab_size = len(tokenizer.word_index) + 1
num_classes = len(label_encoder.classes_)

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_seq_length))
model.add(Conv1D(256, 5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(64, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, epochs=13, batch_size=64, validation_data=(X_test, y_test))
# 进行预测
y_pred = model.predict(X_test)
y_pred_classes = y_pred.argmax(axis=-1)  # 找到概率最高的类别作为预测结果

print("Classification Report:")
print(classification_report(y_test, y_pred_classes))
print("Accuracy:", accuracy_score(y_test, y_pred_classes))