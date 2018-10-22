import os
import sys
import pickle
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import load_model
from prepare_training_data import get_numerical_test_data, get_training_sentences

def get_word_vectors(model, X):
    model_input_sentence = model.layers[0].input
    word_vector_layer_output = model.layers[0].output

    model_calc_word_vectors = K.function([model_input_sentence], [word_vector_layer_output])
    return model_calc_word_vectors([X])[0]


print('Loading word vectors...')
embeddings_index = {}
f = open(os.path.join('./data/glove.6B', 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Done')


def get_closest_word(word_vector):
    closest_distance = 1e9
    closest_word = None

    for word in embeddings_index:
        distance = np.linalg.norm(embeddings_index[word] - word_vector)
        if distance < closest_distance:
            closest_distance = distance
            closest_word = word
        
    return closest_word


pickle_path = './saved_data/10000sentences/'

print('Loading data...')
reddit_pickle_path = os.path.join(pickle_path, 'reddit_sentences.p')
brown_pickle_path = os.path.join(pickle_path, 'brown_sentences.p')

with open(reddit_pickle_path, 'rb') as f_reddit_sentences:
    lst_reddit_sentences = pickle.load(f_reddit_sentences)
with open(brown_pickle_path, 'rb') as f_brown_sentences:
    lst_brown_sentences = pickle.load(f_brown_sentences)
print('Done')

print('Loading model...')
model = load_model(os.path.join(pickle_path, 'model.h5'))
print('Done')

test_sentences = [
    'Way more than that is my guess.'
    # 'Just go to the doctor, you clearly need to.',
    # 'I ate a big bowl of cereal then took a nap.'
]

X_test = get_numerical_test_data(lst_reddit_sentences, lst_brown_sentences, test_sentences, pickle_path)

# get the word vectors first
sentence_word_vectors = get_word_vectors(model, X_test)

# all the word changing stuff
word_vector_input = model.layers[1].input

# get all of the scores for the formal sentence, want to maximize these
model_output_pair = model.layers[-1].output
formality_score_output = tf.gather(model_output_pair, 1, axis=1)

# calculate the gradient of the word vector to change while trying to maximize
# the formality of the entire sentence
formality_maximize_loss = K.sum(formality_score_output)
word_vector_gradients = K.gradients(formality_maximize_loss, word_vector_input)[0]

calculate_word_gradients = K.function([word_vector_input], [word_vector_gradients])

# continuously update the word vector
learning_rate = 10000
word_index = 5
for i in range(0, 30):
    gradients = calculate_word_gradients([sentence_word_vectors])[0]
    sentence_gradient = gradients[0]

    sentence_word_vectors[0] += learning_rate * sentence_gradient

    words = []
    scores = []
    for i, word_vector in enumerate(sentence_word_vectors[0]):
        word = get_closest_word(word_vector)
        words.append(word)

        score = sentence_gradient[i].mean()
        score_padded = str(score).ljust(len(word))
        scores.append(score_padded)
    
    print(' '.join(words))
    print(' '.join(scores))

print('Done')