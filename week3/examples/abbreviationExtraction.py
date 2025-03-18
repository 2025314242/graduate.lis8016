import treform as ptm

s_sent = '최순실 씨가 외국인투자촉진법 (외투촉) 개정안 통과와 예산안 반영까지 꼼꼼이 챙긴 건데, 이른바 외촉법, 어떤 법이길래 최 씨가 열심히 챙긴 걸까요. 자신의 이해관계와 맞아 떨어지는 부분이 없었는지 취재기자와 한걸음 더 들여다보겠습니다. 이서준 기자, 우선 외국인투자촉진법 개정안, 어떤 내용입니까?'
abb_extractor = ptm.abbreviations.SchwartzHearstAbbreviationExtraction()
pairs = abb_extractor(s_sent)
for short, long in pairs.items():
    print(short + " : " + long)
