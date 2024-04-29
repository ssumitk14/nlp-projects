from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag


def extract_nouns(text):
    words = word_tokenize(text)
    tagged_words = pos_tag(words)
    noun_list = [noun for noun, word_tag in tagged_words if word_tag.startswith('NN')]
    return noun_list


def extract_verbs(text):
    words = word_tokenize(text)
    tagged_words = pos_tag(words)
    verb_list = [verb for verb, word_tag in tagged_words if word_tag.startswith('VB')]
    return verb_list


text = """Contrary to popular belief, Lorem Ipsum is not simply random text. 
It has roots in a piece of classical Latin literature from 45 BC, making it over 2000 years old."""

noun_list = extract_nouns(text)
verb_list = extract_verbs(text)

print("Nouns :: ", noun_list)
print("Verbs :: ", verb_list)