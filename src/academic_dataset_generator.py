import os
import nltk
from src import utils


def sentence_generator_from_directory(directory_path, file_limit=None):
    file_counter = 0
    for dirent in os.listdir(directory_path):
        if file_limit is not None and file_counter >= file_limit:
            break

        file_path = os.path.join(directory_path, dirent)
        if not os.path.isfile(file_path):
            continue

        with open(file_path,) as academic_file:
            for line in academic_file:
                cleaned_line = line.strip()

                # skip blank lines
                if len(cleaned_line) == 0:
                    continue

                # skip lines with more/less than two tab separated items, there shouldn't be lines like this
                line_components = cleaned_line.split('\t')
                if len(line_components) != 2:
                    continue

                sentence = line_components[1]
                words = nltk.word_tokenize(sentence)
                yield words

        file_counter += 1


if __name__ == '__main__':
    # generator = sentence_generator_from_directory('./academic_data', file_limit=2)
    # print('\n'.join(list(generator)))

    utils.print_word_stats(sentence_generator_from_directory('./data/academic_data'))

