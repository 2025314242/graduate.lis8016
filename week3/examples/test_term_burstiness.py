import treform as ptm

from treform.weighting.term_burstiness import compute_term_burstiness, compute_pure_term_burstiness

dataset = '../sample_data/news_articles_201701_201812.csv'

pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                            ptm.tokenizer.Komoran(),
                            ptm.helper.POSFilter('NN*'),
                            ptm.helper.SelectWordOnly(),
                            ptm.ngram.NGramTokenizer(2, 2),
                            ptm.helper.StopwordFilter(file='../stopwords/stopwordsKor.txt')
                            )

corpus = ptm.CorpusFromFieldDelimitedFileWithYear('../sample_data/sample_dmr_input.txt',doc_index=2,year_index=1)
pair_map = corpus.pair_map

corpus = ptm.CorpusFromFieldDelimitedFile('../sample_data/ratings.txt', 1, delimiter='\t')
#corpus = ptm.CorpusFromFieldDelimitedFileForClassification(dataset,delimiter=',',doc_index=4,class_index=0)
result = pipeline.processCorpus(corpus.docs)

#print(result)
print()

print('==  ==')
label_list = []
documents = []
for i, doc in enumerate(result):
    document = ' '
    for sent in doc:
        for word in sent:
            if len(word) > 0:
                document += ' ' + word
    document = document.strip()
    if len(document) > 0:
       documents.append(document)
    _label = corpus.pair_map[i]
    label_list.append(_label)

compute_pure_term_burstiness(documents,label_list)