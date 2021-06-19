from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.datasets import imdb

(X_train, y_train), (X_test, y_test)= imdb.load_data(num_words=20000)

print(X_train[0])
print(y_train[0])

X_train= sequence.pad_sequences(X_train, maxlen=80)
X_test= sequence.pad_sequences(X_test, maxlen=80)

model= Sequential()
model.add(Embedding(20000, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=2, validation_data=(X_test, y_test))

score, accuracy= model.evaluate(X_test, y_test, batch_size=32,verbose=2)
print('Test score is : ', score)
print('Model accuracy is : ', accuracy)