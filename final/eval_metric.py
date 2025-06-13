import sacrebleu                 # BLEU
from konlpy.tag import Mecab     # 한국어 형태소 토크나이저 (≈ NLTK 역할)
from evaluate import load        # BERTScore
from tqdm import tqdm
from typing import List


# ────────────────────── 1.  BLEU  ──────────────────────
def BLEU_ko(target: List[str], output: List[str]) -> List[float]:
    """
    sacrebleu for Korean: tokenize='ko-mecab'  (3.x 버전 이상에서 지원)
    → mecab-ko 사전이 설치돼 있어야 함.
      설치 불가하면 tokenize='char' 로 대체해도 무방.
    """
    refs = [[t] for t in target]
    refs = list(zip(*refs))
    token = "ko-mecab"   # or "char"

    scores = []
    for hyp in tqdm(output, desc="BLEU"):
        s = sacrebleu.corpus_bleu([hyp], references=[r for r in refs],
                                  tokenize=token, lowercase=False).score
        scores.append(s)
    return scores


# ────────────────────── 2.  ROUGE-L  ──────────────────────
mecab = Mecab()                   # Konlpy 형태소 분석기

def ko_tokens(sent: str) -> List[str]:
    """형태소 기반 토큰화(스페이스 토큰화보다 정확)"""
    return mecab.morphs(sent)

def LCS(a, b):
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m):
        for j in range(n):
            dp[i+1][j+1] = dp[i][j]+1 if a[i]==b[j] else max(dp[i][j+1], dp[i+1][j])
    return dp[m][n]

def ROUGE_L_ko(target: List[str], output: List[str]) -> List[float]:
    scores = []
    for ref, hyp in tqdm(zip(target, output), total=len(output), desc="ROUGE-L"):
        r_tok, h_tok = ko_tokens(ref), ko_tokens(hyp)
        lcs = LCS(r_tok, h_tok)
        prec = lcs/len(h_tok) if h_tok else 0
        rec  = lcs/len(r_tok) if r_tok else 0
        f1   = 2*prec*rec/(prec+rec) if prec+rec else 0
        scores.append(f1)
    return scores


# ────────────────────── 3.  METEOR  ──────────────────────
"""
NLTK METEOR는 영어 WordNet 기반이라 한국어를 직접 지원하지 않습니다.
간단히 ‘공백 단위 토큰’에 대해 원본 공식을 그대로 쓰거나,
다국어 버전을 별도로 설치해야 합니다 (e.g., json-meteor).
여기서는 **형태소 토큰 + 영문식 가중치**로 근사한 예입니다.
"""
from nltk.translate.meteor_score import meteor_score

def METEOR_ko(target: List[str], output: List[str]) -> List[float]:
    scores = []
    for ref, hyp in tqdm(zip(target, output), total=len(output), desc="METEOR"):
        ref_tok = ko_tokens(ref)
        hyp_tok = ko_tokens(hyp)
        scores.append(meteor_score([ref_tok], hyp_tok))  # ✅ token list로 입력
    return scores


# ────────────────────── 4.  BERTScore  ──────────────────────
def BERTScore_ko(target: List[str], output: List[str], device="cuda") -> List[float]:
    scorer = load("bertscore")
    batch = 32
    out = []
    for i in tqdm(range(0, len(output), batch), desc="BERTScore"):
        preds = output[i:i+batch]
        refs  = target[i:i+batch]
        res = scorer.compute(predictions=preds,
                             references=refs,
                             lang="ko",          # 한국어
                             device=device)
        out.extend(res["f1"])
    return out


import pandas as pd
import torch                           # BERTScore device 확인용

# === 0.  측정 함수 불러오기 =========================
# (위에서 작성한 BLEU_ko, ROUGE_L_ko, METEOR_ko, BERTScore_ko
#   네 함수를 같은 파일에 두거나 import 해 두세요)

model = "TBD"

# ---------------------------------------------------
csv_in  = f"buffer/{model}.csv"        # 원본
csv_out = f"results/{model}.csv"          # 저장
device   = "cuda" if torch.cuda.is_available() else "cpu"

# === 1.  데이터 읽기 ===============================
df = pd.read_csv(csv_in)
preds  = df["pred"].astype(str).tolist()
golds  = df["gold"].astype(str).tolist()

# === 2.  지표 계산 ================================
print("➡ BLEU-ko")
bleu  = BLEU_ko(golds, preds)

print("➡ ROUGE-L-ko")
rouge = ROUGE_L_ko(golds, preds)

print("➡ METEOR-ko")
meteor = METEOR_ko(golds, preds)

print("➡ BERTScore-ko")
berts = BERTScore_ko(golds, preds, device=device)

# === 3.  결과 합치고 저장 =========================
df["bleu"]      = bleu
df["rouge_l"]   = rouge
df["meteor"]    = meteor
df["bertscore"] = berts

df.to_csv(csv_out, index=False)
print(f"✔ 저장 완료: {csv_out} ({len(df)} 행)")