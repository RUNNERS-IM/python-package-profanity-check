from konlpy.tag import Okt
import re

# 한국어 불용어 목록 정의
stop_words = ['이', '그', '저', '것', '수', '있', '하', '되', '않', '같다', '같은', '야']

okt = Okt()


def preprocess_text(text):
    text = text.lower()
    # 숫자 제거
    text = re.sub(r'\d+', '', text)
    # 특수문자 제거
    text = re.sub(r'[^\w\s]', '', text)
    words = okt.morphs(text, stem=True)  # 형태소 분석
    words = [word for word in words if word not in stop_words]  # 불용어 제거
    return ' '.join(words)
