import treform as ptm
import time
from collections import Counter

mecab_path='C:\\mecab\\mecab-ko-dic'
user_dictionary = 'abs.txt'
komoran = ptm.tokenizer.Komoran(userdic=user_dictionary)
