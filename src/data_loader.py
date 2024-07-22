import os

import pandas as pd

from src.text_preprocessing import preprocess_text


def load_data(file_path):
    # 오류가 발생하는 행을 건너뛰도록 설정
    return pd.read_csv(file_path, delimiter='|', header=None, names=['text', 'label'], on_bad_lines='skip')


def preprocess_data(data):
    data['label'] = pd.to_numeric(data['label'], errors='coerce')  # 숫자로 변환, 변환 불가 값은 NaN으로 처리
    data = data.dropna(subset=['label', 'text'])  # NaN 값 제거
    data.loc[:, 'label'] = data['label'].astype(int)
    data.loc[:, 'text'] = data['text'].astype(str)  # 텍스트를 문자열로 변환
    data.loc[:, 'text'] = data['text'].apply(preprocess_text)  # 전처리 적용
    print(data.head())  # 추가: 전처리된 데이터를 출력하여 확인
    return data
