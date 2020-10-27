from autocorrect import Speller
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import numpy as np
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import re
import threading, math

def read_file(fname):
    with open(fname) as f:
        content = f.readlines()
    return [x.strip() for x in content]

def load_w2v():
    global embedding
    lines = read_file("embeddings")
    names = lines[::2]
    vals  = lines[1::2]
    vals = list(map(lambda val : list(map(lambda x : float(x), val.split())), vals))
    embedding = dict(zip(names, vals))

def save_w2v(path_to_w2v, words):
    wv_from_bin = KeyedVectors.load_word2vec_format(datapath(path_to_w2v), binary=True)
    with open("embeddings", 'w') as fd:
        for word in words:
            if word in wv_from_bin:
                vector = wv_from_bin[word]
            else:
                vector = np.zeros(300) + 0.111
            fd.write(word)
            fd.write("\n")
            for num in vector:
                fd.write(str(num))
                fd.write(" ")
            fd.write("\n")

def get_awe(words):
    vector = np.zeros(300)
    num = 0
    for word in words:
        if word in embedding:
            vector += embedding[word]
            num += 1
    vector /= num
    return vector

def StemCorrect(toks):
    stemmer = PorterStemmer()
    for i in range(len(toks)):
        tmp = toks[i]
        toks[i] = stemmer.stem(tmp)
    return toks

def removeStopWords(toks):
    manual_stop = ["w", "rt"]
    new_toks = []
    for word in toks:
        if word in stopwords.words('english') or word in manual_stop:
            continue
        new_toks.append(word)
    return new_toks

def cleanTextString(text):
    text = re.sub('[^A-Za-z]', ' ', text)
    text = text.lower()
    toks = removeStopWords(word_tokenize(text))
    return " ".join(toks)

def cv_split(inputs, outputs, test_frac):
    datasize = len(inputs)
    indices = list(range(0, datasize))
    random.shuffle(indices)
    bp = int(math.ceil(datasize * test_frac))
    test_indices = indices[bp:]
    train_indices = indices[:bp]
    train_inputs = [inputs[i] for i in train_indices]
    train_outputs = [outputs[i] for i in train_indices]
    val_inputs = [inputs[i] for i in test_indices]
    val_outputs = [outputs[i] for i in test_indices]
    return train_inputs, train_outputs, val_inputs, val_outputs
