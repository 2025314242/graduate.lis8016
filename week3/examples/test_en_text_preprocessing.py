import treform as ptm

corpus = ptm.CorpusFromFieldDelimitedFile('../sample_data/sampleEng.txt',1)

#pipeline example 1
pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                        ptm.tokenizer.Word(),
                        ptm.helper.StopwordFilter(file='../stopwords/stopwordsEng.txt'),
                        ptm.stemmer.Porter(),
                        ptm.ngram.NGramTokenizer(min=2, ngramCount=3))
result = pipeline.processCorpus(corpus)
print('== Splitting Sentence + Tokenizing + Stopwords Removal + Stemming : Porter ==')
print(result)
print()

#pipeline example 2
pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                        ptm.tokenizer.Word(),
                        ptm.helper.StopwordFilter(file='../stopwords/stopwordsEng.txt'),
                        ptm.stemmer.Lancaster())
result = pipeline.processCorpus(corpus)
print('== Splitting Sentence + Tokenizing + Stopwords Removal + Stemming : Lancaster ==')
print(result)
print()

#pipeline example 3
pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                        ptm.tokenizer.Word(),
                        #ptm.helper.StopwordFilter(file='../stopwords/stopwordsEng.txt'),
                        ptm.tagger.NLTK(),
                        ptm.lemmatizer.WordNet(),
                        ptm.helper.POSFilter('N*|J*'),
                        ptm.helper.SelectWordOnly(),
                        ptm.helper.StopwordFilter(file='../stopwords/stopwordsEng.txt'),
                        ptm.stemmer.Porter())
result = pipeline.processCorpus(corpus)
print(result)
print()