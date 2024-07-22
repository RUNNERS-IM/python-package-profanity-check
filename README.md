# python-profanity-defend
이 프로젝트는 Naive Bayes 분류기를 사용하여 한국어 텍스트에서 욕설을 탐지하고 필터링하는 기능을 구현합니다. 
사용자는 입력된 텍스트 내의 욕설을 감지하고, 욕설을 `*`로 대체하여 클린한 텍스트를 얻을 수 있습니다.
Python 3.11 버전으로 개발되었습니다.
## 설치하기
### 1. 저장소 클론하기
```sh
   git clone https://github.com/RUNNERS-IM/python-profanity-defend.git
   cd python-profanity-defend
 ```  
### 2. 가상환경 생성하기
```sh
python -m venv venv
source venv/bin/activate  # Windows에서는 `venv\Scripts\activate`
```
### 3. 패키지 설치하기
```sh
  pip install -r requirements.txt
 ```  
## 사용하기
### 1. 실행하기
```sh
python main.py
 ```  
### 2. 예제
```python
texts = ["이 나쁜 새끼야", "좋은 아침입니다"]
for text in texts:
    cleaned_text = replace_swear_words(text, model, vectorizer)
    print(f"Original: {text}")
    print(f"Cleaned: {cleaned_text}")
 ```  
### 3. 디렉토리 구조
```
 python-profanity-defend/
│
├── data/
│   ├── dataset.txt           # 원본 데이터셋
│   ├── augmented_data.txt    # 증강된 데이터셋
│   ├── feedback.txt          # 사용자 피드백 데이터셋
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # 데이터 로드 및 전처리
│   ├── data_augmentation.py  # 데이터 증강
│   ├── model_training.py     # 모델 학습 및 평가
│   ├── text_cleaner.py       # 욕설 필터링
│   ├── text_preprocessing.py # 텍스트 전처리
│   ├── feedback.py           # 사용자 피드백 처리
│
├── main.py                   # 메인 실행 파일
├── requirements.txt          # 필요한 패키지 목록
└── README.md                 # 리드미 파일
```
## 수정이 필요한 부분
1. 일부 욕설이 정상적으로 필터링되지 않는 이슈
2. 아직 모델의 신뢰도가 낮은 이슈 : 신뢰도 목표 90% 이상
```
Name: text, dtype: object
Naive Bayes Accuracy: 77.75%
              precision    recall  f1-score   support

           0       0.80      0.88      0.84       801
           1       0.73      0.59      0.65       435

    accuracy                           0.78      1236
   macro avg       0.76      0.73      0.74      1236
weighted avg       0.77      0.78      0.77      1236

Ensemble Accuracy: 82.12%
              precision    recall  f1-score   support

           0       0.85      0.87      0.86       801
           1       0.76      0.73      0.74       435

    accuracy                           0.82      1236
   macro avg       0.81      0.80      0.80      1236
weighted avg       0.82      0.82      0.82      1236

Cross-validation accuracy: 77.94%
```