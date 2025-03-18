import urllib
import urllib.request
import urllib.parse
import bs4
import re
import time
import random
import datetime
import json
import sys

sid = 'C1nC1dQXH8r8iOHxa4F'

def refineSelection(selection, pqid):
    data = urllib.parse.urlencode({
        'update_back2search_link_param': 'yes',
        'parentQid': pqid,
        'SID': sid,
        'product': 'WOS',
        'databaseId': 'WOS',
        'colName': 'WOS',
        'service_mode': 'Refine',
        'search_mode': 'AdvancedSearch',
        'action': 'search',
        'clickRaMore': '결과 범위를 좁히기 위해 화면에서 선택한 설정은 계속해서 \'추가 ...\' 기능을 사용할 경우 기억되지 않습니다.',
        'openCheckboxes': '결과 범위를 좁히기 위해 이 왼쪽 패널에서 선택한 설정은 숨길 경우 기억되지 않습니다.',
        'refineSelectAtLeastOneCheckbox': '결과 범위를 좁히기 위해 확인란을 최소 1개 선택하십시오.',
        'queryOption(sortBy)': 'PY.D;LD.D;SO.A;VL.D;PG.A;AU.A',
        'queryOption(ss_query_language)': 'auto',
        'sws': '',
        'defaultsws': '결과 내에서 검색...',
        'swsFields': 'TS',
        'swsHidden': '처음 100,000개의 결과 내에서<br>검색',
        'exclude': '',
        'exclude': '',
        'exclude': '',
        'refineSelection': selection,
        'exclude': '',
        'exclude': '',
        'exclude': '',
        'exclude': '',
        'exclude': '',
        'exclude': '',
        'exclude': '',
        'exclude': '',
        'exclude': '',
        'exclude': '',
        'exclude': '',
        'exclude': '',
        'exclude': '',
        'mode': 'refine',
    })
    req = urllib.request.Request('http://apps.webofknowledge.com/Refine.do',
                                 data=data.encode('utf-8'))
    f = urllib.request.urlopen(req)


qid = 1
for year in range(2013, 2018):
    mFrom = 1 + 0
    mTo = 500 + 0
    print('Year :', year)
    refineSelection('PublicationYear_%d' % year, 2)
    qid += 1
    print('C_QID(year %d) %d' %(year, qid))
    nqid = qid
    while mTo < 100000:
        print('Collect (%d, %d)...' % (mFrom, mTo))
        data = urllib.parse.urlencode({
            'selectedIds': '',
            'displayCitedRefs': 'true',
            'displayTimesCited': 'true',
            'displayUsageInfo': 'true',
            'viewType': 'summary',
            'product': 'WOS',
            'rurl': 'http%3A%2F%2Fapps.webofknowledge.com%2Fsummary.do%3Bjsessionid%3DB5F10723A92AC757915DCE55AF2982A1%3Fproduct%3DWOS%26search_mode%3DAdvancedSearch%26doc%3D1%26qid%3D2%26SID%3DC3DNcL7Gs812N5c9CiN',
            'mark_id': 'WOS',
            'colName': 'WOS',
            'search_mode': 'AdvancedSearch',
            'locale': 'ko_KR',
            'view_name': 'WOS-summary',
            'sortBy': 'PY.D;LD.D;SO.A;VL.D;PG.A;AU.A',
            'mode': 'OpenOutputService',
            'qid': nqid,
            'SID': sid,
            'format': 'saveToFile',
            'filters': 'PMID+AUTHORSIDENTIFIERS+ACCESSION_NUM+ISSN+CONFERENCE_SPONSORS+ABSTRACT+CONFERENCE_INFO+SOURCE+TITLE+AUTHORS++',
            'count_new_items_marked': '0',
            'use_two_ets': 'false',
            'IncitesEntitled': 'no',
            'value(record_select_type)': 'range',
            'fields_selection': 'PMID+AUTHORSIDENTIFIERS+ACCESSION_NUM+ISSN+CONFERENCE_SPONSORS+ABSTRACT+CONFERENCE_INFO+SOURCE+TITLE+AUTHORS++',
            'save_options': 'tabWinUTF8',
           'markFrom': mFrom,
           'markTo': mTo,
           'mark_to': mTo,
           'mark_from': mFrom,
                                       })
        req = urllib.request.Request('http://apps.webofknowledge.com/OutboundService.do?action=go&&', data=data.encode('utf-8'))
        #req.add_header('Referer', 'http://apps.webofknowledge.com/Search.do?product=WOS&SID=X1Tg5tVRELhiRHczyzF&search_mode=GeneralSearch&prID=6748dd35-4130-443b-9630-8e82d8463ca3')
        #req.add_header('Origin', 'http://apps.webofknowledge.com')
        #req.add_header('Host', 'apps.webofknowledge.com')
        #req.add_header('User-Agent',
        #               'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36')
        f = urllib.request.urlopen(req)
        cont = f.read().decode('utf-8')
        if cont.find('<!DOCTYPE html>') >= 0:
            print("out of boundary")
            break
        with open('download/download_%d_%d_%d.txt' % (year, mFrom, mTo), 'w', encoding='utf-8') as o:
            o.write(cont)
        mFrom += 500
        mTo += 500
        time.sleep(random.randint(2, 7))
        qid += 1
        print('C_QID', qid)

