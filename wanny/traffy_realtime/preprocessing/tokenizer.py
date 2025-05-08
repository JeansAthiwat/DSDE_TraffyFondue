from pythainlp.tokenize import word_tokenize
from pythainlp.corpus.common import thai_stopwords

def thai_tokenizer(text):
    return word_tokenize(text, engine="newmm")