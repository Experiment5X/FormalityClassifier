import re
import nltk
import operator
import functools
import numpy as np


def load_glove_embeddings(glove_file_path):
    embeddings_index = {}
    f = open(glove_file_path, encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float32')
        except:
            print(word)
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))

    return embeddings_index


def clean_text(text):
    # remove any links
    cleaned_text = re.sub(r'\[[^\]]*\]\([^)]*\)|/[^ ]+', '/LINK/', text)
    cleaned_text = re.sub(r'http[s]?://[\w\d.-_/?&%=;]+', '', cleaned_text)

    # remove numbers, make them all the same /NUMBER/ token
    cleaned_text = re.sub(r'([\d]+.)?[\d]+(st|nd|rd|th)?', '/NUMBER/', cleaned_text)

    # remove -- and *, which is for reddit style formatting
    cleaned_text = re.sub(r'--|\*', '', cleaned_text)

    return cleaned_text


def print_sentences(sentences, limit=100):
    sentence_counter = 0
    for sentence in sentences:
        print(' '.join(sentence))

        sentence_counter += 1
        if sentence_counter >= limit:
            return


def count_words_with_one_occurrence(freq_dist):
    count = 0
    for word in freq_dist.keys():
        if freq_dist[word] == 1:
            count += 1

    return count


def print_word_stats(sentence_generator):
    sentences = list(sentence_generator)
    words = functools.reduce(operator.add, sentences)
    print('Sentence count: ' + str(len(sentences)))
    print('Word count: ' + str(len(words)))

    dist = nltk.FreqDist(words)
    print('Single occurrence words: ' + str(count_words_with_one_occurrence(dist)))


def remove_uncommon_words(freq_dist, sentences, freq_threshold=1, replace_token='/UNCOMMON_WORD/'):
    for sentence in sentences:
        processed_sentence = []
        for word in sentence:
            if not (word in freq_dist) or freq_dist[word] <= freq_threshold:
                processed_sentence.append(replace_token)
            else:
                processed_sentence.append(word)

        yield processed_sentence
