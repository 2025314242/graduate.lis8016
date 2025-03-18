import treform as ptm

'''
collector 모듈에는
* 국내 언론사: 조선일보, 동아일보, 중앙일보, BigKinds, 네이버 뉴스 수집기
* 블로그: 네이버 블로그 (selenium 및 PhantomJS 설치 필요)
수집기가 포함되어 있습니다. 과도하게 데이터를 수집할 시 해당 언론사 사이트로부터 IP 차단을 당할 수 있으니 유의하시길 바랍니다.
'''

# 중앙일보 수집기
cl = ptm.collector.ColJoongang()
cl.collect("실험", 'joongang.txt', startDate='20180923', endDate='20180930')

# 동아일보 수집기
cl = ptm.collector.ColDonga()
cl.collect("실험", 'donga.txt', startDate='20180923', endDate='20180930')

# 네이버뉴스 수집기
cl = ptm.collector.ColNaverNews()
cl.collect("실험", 'navernews.txt', startDate='20180923', endDate='20180930')

# 빅카인즈 뉴스 수집기
cl = ptm.collector.ColBigkinds()
cl.collect("실험", 'bigkinds.txt', startDate='20180923', endDate='20180930')

# 조선일보 수집기
cl = ptm.collector.ColChosun()
cl.collect("실험", 'chosun.txt', startDate='20180923', endDate='20180930')

'''
아래의 수집기들은 다음 모듈을 설치해야 작동합니다.

1. pip로 모듈 설치
>> pip install selenium

2. http://phantomjs.org/download.html 에서 OS에 맞는 phantomjs를 다운받기
 압축을 풀어 실행파일(phantomjs.exe)을 소스코드와 동일한 폴더에 위치
'''

# 네이버 블로그 수집기
cl = ptm.collector.ColNaverBlog()
cl.collect("실험", 'naverblog.txt', startDate='20180923', endDate='20180930')
