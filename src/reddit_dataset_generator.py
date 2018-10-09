import nltk
import json
import utils


def sentence_generator_from_file(comments_file_path, sentence_limit=10000, min_sentence_length=5, max_sentence_length=30):
    sentence_counter = 0
    with open(comments_file_path) as file:
        for line in file:
            comment_data = json.loads(line)
            if len(comment_data['selftext']) == 0:
                continue

            comment_text = comment_data['selftext'].strip()
            if comment_text == '[removed]' or comment_text == '[deleted]':
                continue

            cleaned_text = utils.clean_text(comment_text)
            sentences = nltk.sent_tokenize(cleaned_text)

            for sentence in sentences:
                words = nltk.word_tokenize(sentence)

                if len(words) < min_sentence_length or len(words) > max_sentence_length:
                    continue

                yield words

                sentence_counter += 1
                if sentence_limit is not None and sentence_counter >= sentence_limit:
                    return

                # if sentence_counter % 1000 == 0 and sentence_counter != 0:
                #     print('Finished %d sentences' % sentence_counter)


if __name__ == '__main__':
    generator = sentence_generator_from_file('data/reddit_comments.txt', 1000)
    # utils.print_word_stats(generator)

    for sent in list(generator):
        print(' '.join(sent))
