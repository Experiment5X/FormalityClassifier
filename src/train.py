import pickle
from prepare_training_data import get_numerical_training_data, get_training_sentences
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, Activation
from keras.regularizers import l2


reddit_sentences, brown_sentences = get_training_sentences(sentence_limit=10000)
lst_reddit_sentences = list(reddit_sentences)
lst_brown_sentences = list(brown_sentences)

with open('reddit_sentences.p', 'wb') as f_reddit_sentences:
    pickle.dump(lst_reddit_sentences, f_reddit_sentences)
with open('brown_sentences.p', 'wb') as f_brown_sentences:
    pickle.dump(lst_brown_sentences, f_brown_sentences)

X, Y, embeddings_matrix = get_numerical_training_data(lst_reddit_sentences, lst_brown_sentences)

informalCount = 0
formalCount = 0
for sample_index in range(0, Y.shape[0]):
    if Y[sample_index][0] > Y[sample_index][1]:
        informalCount += 1
    else:
        formalCount += 1

print('Informal count: ' + str(informalCount))
print('Formal count: ' + str(formalCount))
print('Total count: ' + str(informalCount + formalCount))


print('X Shape: ' + str(X.shape))
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


