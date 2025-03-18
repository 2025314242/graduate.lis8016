
import treform as ptm
from treform.sentiment.MLSentimentManager import MachineLearningSentimentAnalyzer

sentiAnalyzer = MachineLearningSentimentAnalyzer()

model = sentiAnalyzer.load('sentiment.model')
vectorizer_model = sentiAnalyzer.loadVectorizer(model_name='senti_vectorizer.model')

# same preprocessing module need to be applied
docs = ['오늘은 세상이 참 아름답게 보이네요! 감사합니다',
        '오늘은 비가와서 그런지 매우 우울하다']
predictions = sentiAnalyzer.predict(docs, model, vectorizer_model)
for i, predicted in enumerate(predictions):
    print(predicted + ' '
                      '.0.0for ' + docs[i])