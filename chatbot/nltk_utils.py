import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np


stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tonkenized_sentence, all_words):
    tonkenized_sentence = [stem(word) for word in tonkenized_sentence]
    bag = np.zeros(len(all_words), dtype= np.float32)

    for index, word in enumerate(all_words):
        if word in tonkenized_sentence:
            bag[index] = 1.0
    
    return bag


