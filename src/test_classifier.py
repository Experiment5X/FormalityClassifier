import os
import sys
import pickle
from keras.models import load_model
from prepare_training_data import get_numerical_test_data, get_training_sentences


if len(sys.argv) <= 1:
    pickle_path = './saved_data/10000sentences/'
elif len(sys.argv) == 2:
    pickle_path = sys.argv[1]
else:
    print('Usage: python3 test_classifier.py [saved_data_path]')

# pickle_path = None

print('Loading data...')


if pickle_path is None:
    reddit_sent_gen, brown_sent_gen = get_training_sentences(sentence_limit=10000)
    lst_reddit_sentences = list(reddit_sent_gen)
    lst_brown_sentences = list(brown_sent_gen)

    with open('reddit_sentences.p', 'wb') as f_reddit_sentences:
        pickle.dump(lst_reddit_sentences, f_reddit_sentences)
    with open('brown_sentences.p', 'wb') as f_brown_sentences:
        pickle.dump(lst_brown_sentences, f_brown_sentences)
else:
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
print()
print()


test_sentences = [
   'Way more than that is my guess.',
    'In the 2 minutes I searched Google, I couldn\'t find an estimate on the number of blackjack tables in Vegas.',
    'I ate a big bowl of cereal then took a nap.',
    'Yeah that doesn\'t really seem right.',
    'Wow that is ridiculous, unbelievable.',
    'Just go to the doctor, you clearly need to.',
    'I coulda sworn I saw that happen.',
    'The synthesized material is hard, optically flawless and usually colorless, but may be made in a variety of different colors.',
    'One new tool, for example, recognizes when a user falls and automatically sends a notification to their emergency contact.',
    'This handout will focus on book reviews.',
    'This clearly demonstrates the significant benefit of using context appropriately in natural language (NL) tasks.',
    'It was one of a series of recommendations by the Texas Research League.',
    'State representatives decided Thursday against taking a poll on what kind of taxes Texans would prefer to pay.',
    'Rep. Wesley Roberts of Seminole , sponsor of the poll idea , said that further delay in the committee can kill the bill.',
    'The West Texan reported that he had finally gotten Chairman Bill Hollowell of the committee to set it for public hearing on Feb. 16.',
]
X_test = get_numerical_test_data(lst_reddit_sentences, lst_brown_sentences, test_sentences, pickle_path)

result = model.predict(X_test)

print('Test Sentences:')
for index, sentence in enumerate(test_sentences):
    print(sentence)

    if result[index][0] > result[index][1]:
        print('Informal sentence')
    else:
        print('Formal sentence')
    print()


print()
while True:
    sentence = input('|Sentence Classifier > ')
    X_test = get_numerical_test_data(lst_reddit_sentences, lst_brown_sentences, [sentence], pickle_path)

    result = model.predict(X_test)
    if result[0][0] > result[0][1]:
        print('Informal sentence')
    else:
        print('Formal sentence')
