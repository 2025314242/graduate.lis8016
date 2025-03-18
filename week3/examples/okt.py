import treform as ptm
import io

pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                        ptm.tokenizer.TwitterKorean(),
                        )

corpus = ptm.CorpusFromFile('../sample_data/sampleKor.txt')
result = pipeline.processCorpus(corpus)
print(result)
