
from treform.word2vec.word_embeddings import Word2Vec

word2vec = Word2Vec()
binary=True
#model_file = "korean_wiki_w2v.bin"
model_file = "word2vec_2.bin"
word2vec.load_model(model_file, binary)

#print(word2vec.most_similar(positives=['대한민국', '도쿄'], negatives=['서울'], topn=10))
print('-----------------------------------')

print(word2vec.similar_by_word('정치'))
