# yTextMiner의 파이썬 버전, PyTextMiner를 ptm이라는 이름으로 사용하겠다고 선언합니다
# ptm 역시 파이프라인 구조로 텍스트를 처리합니다.
# 만약 pyTextMiner에 빨간줄이 계속 뜬다면 왼쪽의 Project 트리뷰에서 pyTextMiner가 포함된 폴더를 우클릭하여
# 'Mark Directory as'에서 'Sources Root'를 눌러주도록 합시다.
# 이 패키지가 동작하기 위해서는 konlpy와 nltk라는 라이브러리가 필요합니다. konlpy는 저번에 설치했으므로,
# 이번에는 nltk를 설치해봅시다. pip install nltk로 간단하게 설치하시면 됩니다.
import treform as ptm

# 제일 먼저 할 일은 파이프라인을 생성하는 것입니다. 먼저 ptm.Pipeline 객체를 생성해보도록 하겠습니다.
# 생성자 파라미터로 우리의 텍스트 분석기가 처리해야할 일을 명시할 수 있습니다.
# ptm.splitter.NLTK()는 nltk라는 패키지에 포함된 Sentence Splitter를 사용합니다.
# 즉 아래와 같이 파이프라인은 문서를 문장 단위로 나눠주는 작업만을 수행합니다.
pipeline = ptm.Pipeline(ptm.splitter.NLTK())

# 다음은 분석에 사용할 corpus를 불러오는 일입니다. sampleEng.txt 파일을 준비해두었으니, 이를 읽어와봅시다.
# ptm의 CorpusFromFile이라는 클래스를 통해 문헌집합을 가져올 수 있습니다. 이 경우 파일 내의 한 라인이 문헌 하나가 됩니다.
corpus = ptm.CorpusFromFile('../sample_data/sampleEng.txt')


# 읽어온 문헌을 파이프라인에 넣어 처리해봅시다. pipeline의 processCorpus를 통해 문헌집합을 처리해봅시다.
# 결과는 result에 넣고 그것을 출력해주었습니다.
result = pipeline.processCorpus(corpus)
print('== Splitting Sentence ==')
print(result)
print()

# 이제 좀더 복잡한 일을 해보도록 하겠습니다. 문장을 분리한 뒤 단어 단위로 tokenizing 해봅시다. 새로운 pipeline을 정의하겠습니다.
# ptm.tokenizer.Word()는 공백 및 구두점 등을 기준으로 단어 단위로 문장을 분리하는 역할을 수행합니다.
# 즉 다음 파이프라인은 문서를 splitter.NLTK에 의해 문장 단위로 쪼갠 뒤, tokenizer.Word()에 의해 단어 단위로 쪼개게 됩니다.
pipeline = ptm.Pipeline(ptm.splitter.NLTK(), ptm.tokenizer.Word())

# 마찬가지로 위에서 사용한 corpus를 처리해보겠습니다.
result = pipeline.processCorpus(corpus)
print('== Splitting Sentence + Tokenizing ==')
print(result)
print()

# 단어 단위로 분리했으니 이제 stopwords를 제거하는게 가능합니다. ptm.helper.StopwordFilter를 사용하여 불필요한 단어들을 지워보도록 하겠습니다.
# 그리고 파이프라인 뒤에 ptm.stemmer.Porter()를 추가하여 어근 추출을 해보겠습니다.
# 한번 코드를 고쳐서 ptm.stemmer.Lancaster()도 사용해보세요. Lancaster stemmer가 Porter stemmer와 어떻게 다른지 비교하면 재미있을 겁니다.
pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                        ptm.tokenizer.Word(),
                        ptm.helper.StopwordFilter(file='../stopwords/stopwordsEng.txt'),
                        ptm.stemmer.Porter())
result = pipeline.processCorpus(corpus)
print('== Splitting Sentence + Tokenizing + Stopwords Removal + Stemming : Porter ==')
print(result)
print()

pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                        ptm.tokenizer.Word(),
                        ptm.helper.StopwordFilter(file='../stopwords/stopwordsEng.txt'),
                        ptm.stemmer.Lancaster())
result = pipeline.processCorpus(corpus)
print('== Splitting Sentence + Tokenizing + Stopwords Removal + Stemming : Lancaster ==')
print(result)
print()

# lemmatization을 위해서는 단어의 POS(Part of Speech) 정보가 필요합니다. 단어에 POS를 부착하는 것을 POS Tagging이라고 하며
# ptm.tagger 내의 클래스를 활용해 영어 POS 태깅이 가능합니다. 다음 구문을 실행하기 위해서는 추가적인 nltk 데이터 파일을 받아야합니다.
# 하단의 Python Console에서 (>는 입력하는거 아닙니다.)
# > import nltk
# > nltk.download()
# 라고 입력하면 nltk download 창을 띄울수 있습니다. 여기에서 Model의 averaged_perceptron_tagger, Corpora의 wordnet을 다운받아 주세요.
pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                        ptm.tokenizer.Word(),
                        #ptm.helper.StopwordFilter(file='../stopwords/stopwordsEng.txt'),
                        ptm.tagger.NLTK(),
                        ptm.lemmatizer.WordNet())
result = pipeline.processCorpus(corpus)
print('== Splitting Sentence + Tokenizing + Stopwords Removal + POS Tagging + Lemmatization ==')
print(result)
print()

# 이번에는 좀 더 파이프라인을 길게 만들어서 lemmatization 후 명사와 형용사만 남기고 다른 단어를 모두 제거해버리겠습니다.
# helper.POSFilter는 원하는 품사만을 골라낼 수 있게 도와주는 유틸리티입니다. NN, JJ와 같이 전체 품사 태그를 모두 입력할 수도 있고
# N*처럼 N으로 시작하는 모든 품사를 골라내도록 입력할 수도 있습니다.
# Pos tagging 이후 파이프라인에서는 단어와 품사 정보가 함께 등장하게 되는데, helper.SelectWordOnly는 이 중에서 품사 정보는 빼고 단어 정보만 출력하도록 도와줍니다.
# 결과가 간단해진 것을 확인할 수 있죠?
pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                        ptm.tokenizer.Word(),
                        #ptm.helper.StopwordFilter(file='stopwordsEng.txt'),
                        ptm.tagger.NLTK(),
                        ptm.lemmatizer.WordNet(),
                        ptm.helper.POSFilter('N*', 'J*'),
                        ptm.helper.SelectWordOnly())
result = pipeline.processCorpus(corpus)
print('== Splitting Sentence + Tokenizing + Stemming + Stopwords Removal + Lemmatization + POS Filtering + Show words only ==')
print(result)
print()


corpus = ptm.CorpusFromFile('../sample_data/sampleKor.txt')

# 이번에는 PyTextMiner로 한국어 처리를 해보도록 하겠습니다. 한국어의 교착어적인 특성 및 복잡한 띄어쓰기 규칙 때문에
# 공백 기준으로 단어를 분리하는 것에는 한계가 있어서 형태소 분석기를 사용합니다.
# ptm.tokenizer.Komoran나 ptm.tokenizer.TwitterKorean을 사용해 형태소 분석이 가능합니다.
# 형태소 분석 이후 품사가 NN으로 시작하는 명사들만 추출하고, 단어만 골라내 출력하도록 해봅시다.
pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                        ptm.tokenizer.Komoran(),
                        ptm.helper.POSFilter('NN*'),
                        ptm.helper.SelectWordOnly(),
                        ptm.helper.StopwordFilter(file='../stopwords/stopwordsKor.txt'))


result = pipeline.processCorpus(corpus)
print('== 문장 분리 + 형태소 분석 + 명사만 추출 + 단어만 보여주기 ==')
print(result)
print()



# 마지막으로 빈도 분석을 실시해봅시다. 빈도 분석은 문헌의 내용을 파악하는 가장 기초적이면서도 통계적으로 중요한 기법입니다.
# 문헌 내의 단어 빈도만을 조사할 것이기 때문에 굳이 문장 분리는 필요없으므로 파이프라인 앞쪽의 ptm.splitter.NLTK()는 빼버렸습니다.
# ptm.counter.WordCounter()를 통해서 빈도 분석이 가능합니다.
pipeline = ptm.Pipeline(ptm.tokenizer.Komoran(),
                        ptm.helper.POSFilter('NN*'),
                        ptm.helper.SelectWordOnly(),
                        ptm.helper.StopwordFilter(file='../stopwords/stopwordsKor.txt'),
                        ptm.counter.WordCounter())
result = pipeline.processCorpus(corpus)
print('== 형태소 분석 + 명사만 추출 + 단어만 보여주기 + 빈도 분석 ==')
print(result)
print()

# 참 쉽죠?