import io
import array
import collections
import numpy as np
import multiprocessing
from time import time
import gensim
from gensim.models.word2vec import LineSentence
from gensim.models import FastText

from gensim.models import Word2Vec


class WordEmbeddings:
    def __init__(self):
        self.documents = []

    def load_model(self, model_file, binary=True):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=binary, unicode_errors='ignore')

    def most_similar(self, positives, negatives, topn=10):
        return self.model.most_similar(positive=positives, negative=negatives, topn=topn)

    def similar_by_word(self, word):
        return self.model.similar_by_word(word)

class GloVe(WordEmbeddings):
    import numpy as np
    import io
    def __init__(self):
        print('GloVe')
        super().__init__()

    def preprocessing(self):
        print('not implemented')

    def train(self):
        print('not implemented')

    def load_model(self, model_file):
        dct = {}
        vectors = array.array('d')

        # Read in the data.
        with io.open(model_file, 'r', encoding='utf-8') as savefile:
            for i, line in enumerate(savefile):
                tokens = line.split(' ')

                word = tokens[0]
                entries = tokens[1:]

                dct[word] = i
                vectors.extend(float(x) for x in entries)

        # Infer word vectors dimensions.
        no_components = len(entries)
        no_vectors = len(dct)

        # Set up the model instance.
        self.no_components = no_components
        self.word_vectors = (np.array(vectors)
                             .reshape(no_vectors,
                                      no_components))
        self.word_biases = np.zeros(no_vectors)
        self.add_dictionary(dct)

    def add_dictionary(self, dictionary):
        """
        Supply a word-id dictionary to allow similarity queries.
        """
        if self.word_vectors is None:
            raise Exception('Model must be fit before adding a dictionary')

        if len(dictionary) > self.word_vectors.shape[0]:
            raise Exception('Dictionary length must be smaller '
                            'or equal to the number of word vectors')

        self.dictionary = dictionary
        if hasattr(self.dictionary, 'iteritems'):
            # Python 2 compat
            items_iterator = self.dictionary.iteritems()
        else:
            items_iterator = self.dictionary.items()

        self.inverse_dictionary = {v: k for k, v in items_iterator}

    def _similarity_query(self, word_vec, number):

        dst = (np.dot(self.word_vectors, word_vec)
               / np.linalg.norm(self.word_vectors, axis=1)
               / np.linalg.norm(word_vec))
        word_ids = np.argsort(-dst)

        return [(self.inverse_dictionary[x], dst[x]) for x in word_ids[:number]
                if x in self.inverse_dictionary]

    def most_similar(self, word, topn=10):
        """
        Run a similarity query, retrieving number
        most similar words.
        """

        if self.word_vectors is None:
            raise Exception('Model must be fit before querying')

        if self.dictionary is None:
            raise Exception('No word dictionary supplied')

        try:
            word_idx = self.dictionary[word]
        except KeyError:
            raise Exception('Word not in dictionary')

        return self._similarity_query(self.word_vectors[word_idx], topn)[1:]

    def most_similars(self, positives, negatives, topn=10):
        """
        Run a similarity query, retrieving number
        most similar words.
        """

        if self.word_vectors is None:
            raise Exception('Model must be fit before querying')

        if self.dictionary is None:
            raise Exception('No word dictionary supplied')

        try:
            #print(str(self.word_vectors.shape))
            embeddings = np.zeros(self.word_vectors.shape, "float32")
            idx = 0
            for i in positives[::2]:
                if len(positives) == 1:
                    word_idx = self.dictionary[i]
                    embeddings = self.word_vectors[word_idx]
                else:
                    j = positives[idx+1]
                    word_idx1 = self.dictionary[i]
                    word_idx2 = self.dictionary[j]
                    embeddings = np.add(self.word_vectors[word_idx1], self.word_vectors[word_idx2])
                    idx += 2;
            for i in negatives:
                word_idx = self.dictionary[i]
                embeddings = np.subtract(embeddings, self.word_vectors[word_idx])

        except KeyError:
            raise Exception('Word not in dictionary')

        return self._similarity_query(embeddings, topn)[1:]


if 'name' == '__main__':
    glove = GloVe()
    binary = True
    model_file = '../glove-win_devc_x64/vectors.txt'
    glove.load_model(model_file)
    print(glove.most_similars(positives=['이재명'], negatives=[], topn=10))

    print('-----------------------------------')

    print(glove.most_similar('이재명'))