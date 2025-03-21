import sys
import tomotopy as tp
import itertools
import numpy as np
import pyLDAvis

input_file = "../sample_data/3_class_naver_news.csv"

corpus = tp.utils.Corpus()
for line in open(input_file, encoding='utf-8'):
    fd = line.strip().split(',')
    if len(fd) < 4:
        continue

    time_stamp = fd[0]
    section = fd[1]

    text = fd[3] + " " + fd[4]
    corpus.add_doc(text.split(), multi_metadata=['y_' + time_stamp, 's_' + section])

mdl = tp.DMRModel(tw=tp.TermWeight.ONE,
                      k=20,
                      corpus=corpus
                      )
mdl.optim_interval = 20
mdl.burn_in = 200

mdl.train(0)

print('Num docs:{}, Num Vocabs:{}, Total Words:{}'.format(
    len(mdl.docs), len(mdl.used_vocabs), mdl.num_words
))

# Let's train the model
for i in range(0, 100, 20):
    print('Iteration: {:04} LL per word: {:.4}'.format(i, mdl.ll_per_word))
    mdl.train(20)
print('Iteration: {:04} LL per word: {:.4}'.format(2000, mdl.ll_per_word))

mdl.summary()


year_labels = sorted(l for l in mdl.multi_metadata_dict if l.startswith('y_'))
section_labels = sorted(l for l in mdl.multi_metadata_dict if l.startswith('s_'))

# calculate topic distribution with each metadata using get_topic_prior()
print('Topic distributions by year')
for l in year_labels:
    print(l, '\n', mdl.get_topic_prior(multi_metadata=[l]), '\n')

print('Topic distributions by section')
for l in section_labels:
    print(l, '\n', mdl.get_topic_prior(multi_metadata=[l]), '\n')

# Also we can estimate topic distributions with multiple metadata
print('Topic distributions by year-journal')
for y, j in itertools.product(year_labels, section_labels):
    print(y, ',', j, '\n', mdl.get_topic_prior(multi_metadata=[y, j]), '\n')


for d in mdl.docs:
    print(d.get_topic_dist())

#for pyLDAVIS
topic_term_dists = np.stack([mdl.get_topic_word_dist(k) for k in range(mdl.k)])
doc_topic_dists = np.stack([doc.get_topic_dist() for doc in mdl.docs])
doc_topic_dists /= doc_topic_dists.sum(axis=1, keepdims=True)
doc_lengths = np.array([len(doc.words) for doc in mdl.docs])
vocab = list(mdl.used_vocabs)
term_frequency = mdl.used_vocab_freq

prepared_data = pyLDAvis.prepare(
    topic_term_dists,
    doc_topic_dists,
    doc_lengths,
    vocab,
    term_frequency,
    start_index=0, # tomotopy starts topic ids with 0, pyLDAvis with 1
    sort_topics=False # IMPORTANT: otherwise the topic_ids between pyLDAvis and tomotopy are not matching!
)
pyLDAvis.save_html(prepared_data, 'ldavis.html')