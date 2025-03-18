from treform.topic_model.pyTextMinerTopicModel import pyTextMinerTopicModel
import treform as ptm
import tomotopy as tp

if __name__ == '__main__':

    mecab_path='C:\\mecab\\mecab-ko-dic'
    pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                            ptm.tokenizer.Komoran(),
                            ptm.helper.POSFilter('NN*'),
                            ptm.helper.SelectWordOnly(),
                            ptm.helper.StopwordFilter(file='../stopwords/stopwordsKor.txt')
                            )

    #corpus = ptm.CorpusFromFieldDelimitedFileWithYear('../sample_data/sample_dmr_input.txt',doc_index=2,year_index=1)
    #pair_map = corpus.pair_map

    corpus = ptm.CorpusFromFieldDelimitedFile('../sample_data/ratings.txt', 1, delimiter='\t')
    #print(corpus.docs[10])

    #pair_map = None

    result = pipeline.processCorpus(corpus.docs[:1000])
    text_data = []
    for doc in result:
        new_doc = []
        for sent in doc:
            for _str in sent:
                if len(_str) > 0:
                    new_doc.append(_str)
        if len(new_doc) > 0:
            print(new_doc)
            text_data.append(new_doc)

    topic_model = pyTextMinerTopicModel()
    topic_number=10
    #dominant_topic_number = 6
    #if dominant_topic_number >= topic_number:
    #    dominant_topic_number = topic_number - 1

    mdl=None
    #mode is either lda, dmr, hdp, infer, ct, visualize, etc

    mode='ct'
    label = ''
    if mode == 'lda':
        print('Running LDA')
        label='LDA'
        lda_model_name = './test.lda.bin'
        mdl=topic_model.lda_model(text_data, lda_model_name, topic_number)

        print('perplexity score ' + str(mdl.perplexity))

    elif mode == 'dmr':
        print('Running DMR')
        label='DMR'
        dmr_model_name='./test.dmr.bin'
        mdl=topic_model.dmr_model(text_data, pair_map, dmr_model_name, topic_number)

        print('perplexity score ' + str(mdl.perplexity))

    elif mode == 'hdp':
        print('Running HDP')
        label='HDP'
        hdp_model_name='./test.hdp.bin'
        mdl, topic_num=topic_model.hdp_model(text_data, hdp_model_name)
        topic_number=topic_num

        print('perplexity score ' + str(mdl.perplexity))

    elif mode == 'hlda':
        print('Running HLDA')
        label='HLDA'
        hlda_model_name = './test.hlda.bin'
        mdl=topic_model.hlda_model(text_data, hlda_model_name)
        print('perplexity score ' + str(mdl.perplexity))

    elif mode == 'ct':
        print('Running CT')
        label = 'CT'
        ct_model_name = './test.ct.bin'
        save_file = './topic_network.html'
        mdl = topic_model.ct_model(text_data, ct_model_name, topic_number=topic_number, topic_network_result=save_file)

    elif mode == 'infer':
        lda_model_name = './test.lda.bin'
        unseen_text='피부 영양 수분 유지하다 많다 도움_주다 하다 출처'
        topic_model.inferLDATopicModel(lda_model_name, unseen_text)

    if mode == 'visualize':
        model_name = './test.lda.bin'
        if model_name == './test.lda.bin':
            mdl = tp.LDAModel.load(model_name)
        elif model_name == './test.dmr.bin':
            mdl = tp.DMRModel.load(model_name)
            visual_result_file1= '../dmr_line_graph.png'
            visual_result_file2 = '../dmr_bar_graph.png'
            topic_model.visualizeDMR(mdl,visual_result1=visual_result_file1, visual_result2=visual_result_file2)
        elif model_name == './test.ct.bin':
            mdl = tp.CTModel.load(model_name)
            result_file = 'D:/python_workspace/treform/topic_network.html'
            topic_model.visualize_ct_model(mdl, topic_network_result=result_file)

        mdl.load(model_name)
        # The below code extracts this dominant topic for each sentence
        # and shows the weight of the topic and the keywords in a nicely formatted output.
        df_topic_sents_keywords, matrix = topic_model.format_topics_sentences(topic_number=topic_number, mdl=mdl)


        #visualize documents by tSNE
        topic_model.tSNE(mdl,matrix,label,topic_number=topic_number)

        visualization_file='../topic_visualization.html'
        topic_model.make_pyLDAVis(mdl,visualization_file=visualization_file)
