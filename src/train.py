from src.prepare_training_data import get_numerical_training_data, get_training_sentences
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, Activation
from keras.regularizers import l2


reddit_sentences, brown_sentences = get_training_sentences(sentence_limit=10000)
lst_reddit_sentences = list(reddit_sentences)
lst_brown_sentences = list(brown_sentences)

X, Y, embeddings_matrix = get_numerical_training_data(lst_reddit_sentences, lst_brown_sentences)
print('Finished getting all data')

unique_word_count = embeddings_matrix.shape[0]

model = Sequential()
model.add(Embedding(unique_word_count, 100, input_length=30, weights=[embeddings_matrix]))
model.add(LSTM(500, return_sequences=True))
model.add(LSTM(100, return_sequences=False))
model.add(Dense(2, kernel_regularizer=l2()))
model.add(Activation('softmax'))

model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['categorical_accuracy'])
model.fit(X, Y, batch_size=32, epochs=10, validation_split=0.2, verbose=1)

model.save('model.h5')


