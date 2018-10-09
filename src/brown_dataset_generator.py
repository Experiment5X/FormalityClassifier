import nltk
import utils


def sentence_generator(sentence_limit=10000, min_sentence_length=5, max_sentence_length=30):
    sentence_counter = 0
    for sentence in nltk.corpus.brown.sents():
        if len(sentence) < min_sentence_length or len(sentence) > max_sentence_length:
            continue

        cleaned_sentence = [utils.clean_text(word) for word in sentence]
        yield cleaned_sentence

        sentence_counter += 1
        if sentence_limit is not None and sentence_counter >= sentence_limit:
            return


if __name__ == '__main__':
    generator = sentence_generator(1000)
    utils.print_sentences(generator)
