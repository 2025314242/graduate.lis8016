import MeCab
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from collections import Counter
from tqdm import tqdm
from typing import List


def count_words(
        texts: pd.DataFrame,
        tagger: MeCab.Tagger,
        counter: Counter,
        pos_list: List[str],
        remain: bool=True,
        is_uni_gram: bool=False
    ):
    if is_uni_gram and not remain:
        return count_words_n_gram(texts, tagger, counter, stop_pos_list=pos_list)
    
    for text in tqdm(texts, total=len(texts)):
        node = tagger.parseToNode(text)
        while node:
            word = node.surface
            pos = node.feature.split(",")[0]
            if remain:
                if pos in pos_list:
                    counter[word] += 1
            else:
                if pos not in pos_list:
                    counter[word] += 1
            node = node.next
    
    return counter


def tokenize(text: str, tagger: MeCab.Tagger, stop_pos_list: List[str]):
    node = tagger.parseToNode(text)
    words = []
    while node:
        word = node.surface.strip()
        pos = node.feature.split(",")[0]
        if pos not in stop_pos_list:
            words.append(word)
        node = node.next
    
    return words


def count_words_n_gram(
        texts: pd.DataFrame,
        tagger: MeCab.Tagger,
        counter: Counter,
        stop_pos_list: List[str],
    ):
    for text in tqdm(texts, total=len(texts)):
        tokens = tokenize(text, tagger, stop_pos_list)
        
        for i in range(len(tokens) - 1):
            bigram = tuple(tokens[i:i+2])
            counter[bigram] += 1
        
        for i in range(len(tokens) - 2):
            trigram = tuple(tokens[i:i+3])
            counter[trigram] += 1
    
    return counter


def main(
        num: int,
        df: pd.DataFrame,
        tagger: MeCab.Tagger,
        counter: Counter
    ):
    texts = df['Article'].dropna().astype(str)
    
    if num == 1:
        title = 'only noun'
        pos_list = ['NNG', 'NNP', 'NNB', 'NNBC', 'NR', 'NP']
        counter = count_words(texts, tagger, counter, pos_list)
    elif num == 2:
        title = 'only verb'
        pos_list = ['VV', 'VA', 'VX', 'VCP', 'VCN']
        counter = count_words(texts, tagger, counter, pos_list)
    elif num == 3:
        title = 'all'
        pos_list = []
        counter = count_words(texts, tagger, counter, pos_list, remain=False)
    elif num == 4:
        title = 'bi-/tri-gram'
        pos_list = [
            'MAG', 'MAJ', 'IC',
            'JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JX', 'JC',
            'EP', 'EF', 'EC', 'ETN', 'ETM',
            'XPN', 'XSN', 'XSV', 'XSA',
            'SF', 'SE', 'SS', 'SP', 'SO', 'SW',
            'SL', 'SH', 'SN'
        ]
        counter = count_words(texts, tagger, counter, pos_list, remain=False, is_uni_gram=True)
    else:
        return
    
    freqs = counter.most_common()
    ranks = np.arange(1, len(freqs) + 1)
    frequencies = np.array([freq for _, freq in freqs])
    
    plt.figure(figsize=(10, 6))
    plt.plot(ranks, frequencies)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Rank (log)')
    plt.ylabel('Frequency (log)')
    plt.title(f"Zipf's Law ({title})")
    plt.grid(True)
    plt.savefig(f"{title.replace(' ', '_').replace('-', '_').replace('/', '')}.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=str, required=True, help='number')
    args, _ = parser.parse_known_args()
    
    names = ['Date', 'Topic', 'Broadcasting', 'Article', 'Reference']
    df = pd.read_csv('news_articles_201701_201812.csv', header=None, names=names)
    
    tagger = MeCab.Tagger()
    
    counter = Counter()
    
    main(num=int(args.n), df=df, tagger=tagger, counter=counter)