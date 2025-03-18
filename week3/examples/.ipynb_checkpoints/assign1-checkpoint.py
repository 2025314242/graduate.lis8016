import treform as ptm
import platform
from nltk.probability import FreqDist
import re, operator
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pylab as pl
from wordcloud import WordCloud





'''
komoran_dic = './komoran_dic.txt'
pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                        ptm.tokenizer.Komoran(userdic=komoran_dic),
                        ptm.helper.POSFilter('NN*|V*'),
                        ptm.helper.SelectWordOnly(),
                        ptm.ngram.NGramTokenizer(min=1, ngramCount=2, concat='_'),
                        ptm.helper.StopwordFilter(file='../stopwords/stopwordsKor.txt')
                        )
'''
#no. 1
komoran_dic = './komoran_dic.txt'

execution = 2
if execution == 1:
    pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                            ptm.tokenizer.Komoran(userdic=komoran_dic),
                            ptm.helper.POSFilter('NN*'),
                            ptm.helper.SelectWordOnly(),
                            ptm.counter.WordCounter()
                            )

elif execution == 2:
    #no 2
    pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                            ptm.tokenizer.Komoran(userdic=komoran_dic),
                            ptm.helper.POSFilter('V*'),
                            ptm.helper.SelectWordOnly(),
                            ptm.counter.WordCounter()
                            )
elif execution == 3:
    #no. 3
    pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                            ptm.tokenizer.MaxScoreTokenizerKorean(),
                            ptm.counter.WordCounter())
elif execution == 4:
    #no. 4
    pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                            ptm.tokenizer.Komoran(userdic=komoran_dic),
                            ptm.helper.SelectWordOnly(),
                            ptm.helper.StopwordFilter(file='../stopwords/stopwordsKor.txt'),
                            ptm.ngram.NGramTokenizer(min=2, ngramCount=3, concat='_'),
                            ptm.counter.WordCounter()
                            )


#corpus = ptm.CorpusFromFile('../sample_data/sampleKor.txt')

corpus = ptm.CorpusFromFieldDelimitedFile('../sample_data/news_articles_201701_201812.csv',3,delimiter=",")

#corpus = corpus.docs[:2]

result = pipeline.processCorpus(corpus.docs[:10])

print(len(result))
print(result)

#doc_collection = ''
term_counts = {}
for doc in result:
    for sent in doc:
        for _str in sent:
            term_counts[_str[0]] = term_counts.get(_str[0], 0) + int(_str[1])
            freq = range(int(_str[1]))
            co = ''
            for n in freq:
                co +=  ' ' + _str[0]
            #doc_collection += ' ' + co

term_fdist = FreqDist()
word_freq = []
for key, value in term_counts.items():
    word_freq.append((value,key))
    term_fdist[key] += value

#fontprop = fm.FontProperties(fname=font_path)
#NUM_PLOT = 200
#plotFdist(n=NUM_PLOT, title='Corpus', fdist=wordOnlyFDist(term_fdist), fontprop=fontprop)

word_freq.sort(reverse=True)
print(word_freq)

f = open("demo_result.txt", "w", encoding='utf8')
for pair in word_freq:
    f.write(pair[1] + '\t' + str(pair[0]) + '\n')
f.close()

# Generate a word cloud image
#wordcloud = WordCloud().generate(doc_collection)

