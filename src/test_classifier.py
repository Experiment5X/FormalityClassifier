from keras.models import load_model
from src.prepare_training_data import get_numerical_test_data, get_training_sentences


print('Loading data...')
reddit_sent_gen, brown_sent_gen = get_training_sentences(sentence_limit=10000)

lst_reddit_sentences = list(reddit_sent_gen)
lst_brown_sentences = list(brown_sent_gen)
print('Done')

print('Loading model...')
model = load_model('model.h5')
print('Done')

while True:
    sentence = input('|Sentence Classifier > ')
    X_test = get_numerical_test_data(lst_reddit_sentences, lst_brown_sentences, [sentence])

    result = model.predict(X_test)
    if result[0][0] > result[0][1]:
        print('Informal sentence')
    else:
        print('Formal sentence')
