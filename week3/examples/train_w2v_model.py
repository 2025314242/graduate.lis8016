
from treform.word2vec.word_embeddings import Word2Vec

word2vec = Word2Vec()
mode = 'simple'
mecab_path = 'C:\\mecab\\mecab-ko-dic'
stopword_file = '../stopwords/stopwordsKor.txt'
files = []
files.append('../data/content.txt')
is_directory=False
doc_index=-1
max=-1
is_mecab=False

#mode, path, stopword_file, files, is_directory=False, doc_index=-1, max=-1
word2vec.preprocessing(mode,mecab_path,stopword_file,files,is_directory,doc_index,max)

min_count=5
window=5
size=200
negative=10
word2vec.train(min_count, window, size, negative)

model_file = 'word2vec_2.txt'
binary=False;
word2vec.save_model(model_file, binary)
word2vec.load_model(model_file, binary)

print(word2vec.most_similar(positives=['이승엽', '축구'], negatives=['야구'], topn=10))
