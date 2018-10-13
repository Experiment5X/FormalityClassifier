import os
import math
import pickle
import pymysql
from keras.models import load_model
from prepare_training_data import get_numerical_test_data, get_training_sentences


pickle_path = './saved_data/10000sentences/'

reddit_pickle_path = os.path.join(pickle_path, 'reddit_sentences.p')
brown_pickle_path = os.path.join(pickle_path, 'brown_sentences.p')
tokenizer_pickle_path = os.path.join(pickle_path, 'tokenizer.p')

with open(reddit_pickle_path, 'rb') as f_reddit_sentences:
    lst_reddit_sentences = pickle.load(f_reddit_sentences)
with open(brown_pickle_path, 'rb') as f_brown_sentences:
    lst_brown_sentences = pickle.load(f_brown_sentences)
with open(tokenizer_pickle_path, 'rb') as f_tokenizer:
    tokenizer = pickle.load(f_tokenizer)

words = tokenizer.word_index
words_formality_classifications = {}

# create a new dictionary for storing the number of formal and informal
# sentences that each of the words appears in
# (informal_sentence_count, formal_sentence_count)
# for w in words.keys():
#     words_formality_classifications[w] = (0, 0)

model = load_model(os.path.join(pickle_path, 'model.h5'))

all_sentences = lst_reddit_sentences
all_sentences.extend(lst_brown_sentences)

# classify the reddit sentences
batch_size = 30
batch_count = len(lst_reddit_sentences) // batch_size

for batch_index in range(0, batch_count):
    sentence_batch = lst_reddit_sentences[batch_index * 30: (batch_index + 1) * 30]

    X = get_numerical_test_data(lst_reddit_sentences, lst_brown_sentences, sentence_batch, pickle_path, True)
    Y = model.predict(X)

    for sent_index in range(0, len(sentence_batch)):
        cur_sentence = sentence_batch[sent_index]
        cur_y = Y[sent_index]

        informal_sentence = cur_y[0] > cur_y[1]

        # update the frequencies for all of the words in the current sentence
        for word in cur_sentence:
            if not word.lower() in words_formality_classifications:
                words_formality_classifications[word.lower()] = [0, 0]

            if informal_sentence:
                words_formality_classifications[word.lower()][0] += 1
            else:
                words_formality_classifications[word.lower()][1] += 1

    print('Processed batch %d of %d' % (batch_index + 1, batch_count))


print('Found %d total words' % (len(words_formality_classifications.keys())))

sorted_by_appearance_count = sorted(words_formality_classifications.items(), key=lambda kv: -(kv[1][0] + kv[1][1]))

# grab credentials
with open('db_credentials.txt') as file:
    credentials = [x.strip().split(":") for x in file.readlines()]

for username, password in credentials:
    user = username
    pw = password

# create connection object
conn = pymysql.connect(host='ws-db.cxn6r23mlloe.us-east-1.rds.amazonaws.com', user=user, password=pw, db='corpus')
cursor = conn.cursor()

print('Word\tInformal\tFormal')
for word, frequency in sorted_by_appearance_count[:25]:
    print('%s\t%d\t%d' % (word, frequency[0], frequency[1]))


data = []

for word, frequency in sorted_by_appearance_count:
    # ignore words with non-ascii characters in them
    if not all(ord(c) < 128 for c in word):
        continue

    word_score = float(frequency[1]) / float(frequency[0] + frequency[1])
    data.append((word, word_score))


query = 'INSERT INTO Word_Formality (word,formality) VALUES (%s, %s)'
cursor.executemany(query, data)

conn.commit()
conn.close()
print('Updated database')
