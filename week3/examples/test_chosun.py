import treform as ptm


# 조선일보 수집기
cl = ptm.collector.ColChosun()
cl.collect("대통령", 'chosun.txt', startDate='20180923', endDate='20180930')