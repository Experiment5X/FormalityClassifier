import nltk
import utils
import pandas as pd


# from: https://towardsdatascience.com/how-i-used-machine-learning-to-classify-emails-and-turn-them-into-insights-efed37c1e66
def parse_raw_message(raw_message):
    lines = raw_message.split('\n')
    email = {}
    message = ''
    keys_to_extract = ['from', 'to']
    for line in lines:
        if ':' not in line:
            message += line.strip()
            email['body'] = message
        else:
            pairs = line.split(':')
            key = pairs[0].lower()
            val = pairs[1].strip()
            if key in keys_to_extract:
                email[key] = val
    return email


def sentence_generator_from_file(file_path, sentence_limit=10000):
    emails = pd.read_csv(file_path)

    sentence_count = 0
    for _, email_row in emails.iterrows():
        email_info = parse_raw_message(email_row[1])
        email_message = email_info['body']
        
        cleaned_message = utils.clean_text(email_message)
        sentences = nltk.sent_tokenize(cleaned_message)

        for sent in sentences:
            yield sent

            sentence_count += 1
            if sentence_count != 0 and sentence_count % 1000 == 0:
                print('Found %d sentences' % sentence_count)
            if sentence_count > sentence_limit:
                return


if __name__ == '__main__':
    sentences = list(sentence_generator_from_file('/Users/adamspindler/Downloads/emails.csv', sentence_limit=1e8))
    print(len(sentences))
