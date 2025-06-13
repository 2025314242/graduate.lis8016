import argparse
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
import re
import tomotopy as tp
from konlpy.tag import Okt
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple
from wordcloud import WordCloud


# --- Step 1 --- #

def save_pickle(obj, path: str):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_data(path: str='data/food_news.txt') -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t', header=None, names=['title', 'date', 'source', 'content'], on_bad_lines='skip')
    df = df.dropna(subset=['title', 'date', 'source', 'content']) # remove nan lines
    return df

def tokenize(text: str, okt: Okt, stopwords: List[str]) -> List[str]:
    text = re.sub(r'[^가-힣a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    tokens = okt.nouns(text)
    tokens = [token for token in tokens if token not in stopwords and len(token) > 1]
    return tokens

def preprocess(df: pd.DataFrame) -> Tuple[List[List[str]], List[str], List[str]]:
    okt = Okt()
    CUSTOM_STOPWORDS = list(set([
        '수', '것', '기자', '경우', '내용',
        '관련', '위반', '대상', '확인', '보도',
        '기사', '사진', '뉴스', '전달', '소식',
        '자료', '이번', '이날', '지난해', '올해',
        '중', '후'
    ]))
    
    docs, metas, srcs = [], [], []
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc='Preprocess...'):
        text = (row['title'] or '') + ' ' + (row['content'] or '')
        tokens = tokenize(text, okt=okt, stopwords=CUSTOM_STOPWORDS)
        docs.append(tokens)
        
        date = str(row['date'])
        year = date[:4] if len(date) >= 4 else 'Unknown'
        metas.append(year)
        
        source = str(row['source']) if pd.notna(row['source']) else 'Unknown'
        srcs.append(source)
    return docs, metas, srcs


# --- Step 2 --- #

def train_lda(docs: List[List[str]], num_topics: int):
    model = tp.LDAModel(k=num_topics)
    for doc in tqdm(docs, total=len(docs), desc='Train LDA...'):
        model.add_doc(doc)
    model.train(1000)
    return model

def train_dmr(docs: List[List[str]], metas: List[str], num_topics: int):
    model = tp.DMRModel(k=num_topics)
    for doc, meta in tqdm(zip(docs, metas), total=len(docs), desc='Train DMR...'):
        model.add_doc(doc, metadata=meta)
    model.train(1000)
    return model


# --- Step 3 --- #

def visualize(
    model,
    metas: List[str],
    srcs: List[str],
    type: str,
    num_topics: int,
    top_n: int=30,
    output_dir: str='figures'
):
    output_dir = os.path.join(output_dir, f'{type.lower()}_{num_topics}')
    os.makedirs(output_dir, exist_ok=True)
    
    plt.rcParams['font.family'] = 'NanumGothic'
    
    for k in tqdm(range(model.k), total=model.k, desc='Visualizing...'):
        # Type 1. Wordcloud
        topic_words = model.get_topic_words(k, top_n=top_n)
        word_freq = {word: weight for word, weight in topic_words}
        
        wc = WordCloud(
            font_path='NanumGothic',
            background_color='white',
            width=800,
            height=600
        ).generate_from_frequencies(word_freq)
        
        filepath_wc = os.path.join(output_dir, f'wordcloud_topic_{k}.png')
        plt.figure(figsize=(8, 6))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filepath_wc)
        plt.close()
        
        # Type 2. Trend (over time)
        topic_probs = [doc.get_topic_dist()[k] for doc in model.docs]
        df_time = pd.DataFrame({'year': metas, 'prob': topic_probs})
        df_time = df_time.groupby('year').mean().reset_index()

        plt.figure(figsize=(8, 6))
        plt.plot(df_time['year'], df_time['prob'], marker='o')
        plt.title(f"Topic {k}: Trend over Time")
        plt.xlabel("Year")
        plt.ylabel("Topic Proportion")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        filepath_trend = os.path.join(output_dir, f'trend_topic_{k}.png')
        plt.savefig(filepath_trend)
        plt.close()
        
        # Type 3. Distribution (by source)
        df_source = pd.DataFrame({'source': srcs, 'prob': topic_probs})
        df_source = df_source.groupby('source').mean().reset_index()

        plt.figure(figsize=(10, 6))
        plt.bar(df_source['source'], df_source['prob'])
        plt.title(f"Topic {k}: Distribution by Source")
        plt.xlabel("Source")
        plt.ylabel("Average Topic Proportion")
        plt.xticks(rotation=45)
        plt.tight_layout()
        filepath_source = os.path.join(output_dir, f'source_topic_{k}.png')
        plt.savefig(filepath_source)
        plt.close()


# --- Main --- #

def main(
    model_type: str,
    num_topics_list=[10, 20, 30, 40, 50]
    ):
    # Step 1. Preprocessing data
    if os.path.exists('data/docs.pkl') and os.path.exists('data/metas.pkl') and os.path.exists('data/srcs.pkl'):
        docs = load_pickle('data/docs.pkl')
        metas = load_pickle('data/metas.pkl')
        srcs = load_pickle('data/srcs.pkl')
    else:
        df = load_data()
        docs, metas, srcs = preprocess(df)

        os.makedirs('data', exist_ok=True)
        save_pickle(docs, 'data/docs.pkl')
        save_pickle(metas, 'data/metas.pkl')
        save_pickle(srcs, 'data/srcs.pkl')
    
    for num_topics in num_topics_list:
        # Step 2. Training
        model_path = f'models/{model_type.lower()}_model_{num_topics}.bin'
        model_exists = Path(model_path).exists()
        
        if model_exists:
            if model_type == 'LDA':
                model = tp.LDAModel.load(model_path)
            elif model_type == 'DMR':
                model = tp.DMRModel.load(model_path)
            elif model_type == 'BERTopic':
                pass
            else:
                model = None
        else:
            if model_type == 'LDA':
                model = train_lda(docs, num_topics=num_topics)
            elif model_type == 'DMR':
                model = train_dmr(docs, metas, num_topics=num_topics)
            elif model_type == 'BERTopic':
                pass
            else:
                model = None
            model.save(f'models/{model_type.lower()}_model_{num_topics}.bin')
        
        print(f"=== [{model_type} Topics] ===")
        for k in range(model.k):
            print(f"Topic {k+1}: ", model.get_topic_words(k, top_n=10))
    
        # Step 3. Visualization
        if num_topics > 20: # only for 10 and 20
            exit()
        
        visualize(
            model=model,
            metas=metas,
            srcs=srcs,
            type=model_type,
            num_topics=num_topics
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', type=str, required=True, choices=['LDA', 'DMR', 'BERTopic'])
    args, _ = parser.parse_known_args()
    
    main(
        model_type=args.t
    )