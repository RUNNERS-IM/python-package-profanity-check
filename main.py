import os
import pandas as pd
from src.data_loader import load_data, preprocess_data
from src.data_augmentation import augment_data_parallel
from src.model_training import train_and_evaluate_model
from src.text_cleaner import replace_swear_words
from src.feedback import load_feedback_data

# 데이터 파일 경로
original_data_path = 'data/dataset.txt'
augmented_data_path = 'data/augmented_data.txt'
feedback_file_path = 'data/feedback.txt'


def main():
    if os.path.exists(augmented_data_path):
        # 1. 증강 데이터 로드
        print("Loading augmented data...")
        combined_data = pd.read_csv(augmented_data_path, delimiter='|', names=['text', 'label'])
    else:
        # 1. 원본 데이터 로드 및 전처리
        print("Loading original data...")
        original_data = load_data(original_data_path)
        feedback_data = load_feedback_data(feedback_file_path)

        # 'label' 열이 존재하는지 확인
        if 'label' not in original_data.columns:
            raise KeyError("Original data does not contain 'label' column")
        if 'label' not in feedback_data.columns:
            feedback_data['label'] = 0  # 또는 적절한 기본값 설정

        # 2. 원본 데이터와 피드백 데이터 결합 및 전처리
        combined_data = pd.concat([original_data, feedback_data], ignore_index=True)
        combined_data = preprocess_data(combined_data)
        print(f"Combined data after preprocessing: {combined_data.shape}")

        # 전처리 후 데이터 확인
        print(combined_data.head())

        # 3. 데이터 증강
        augmented_data = augment_data_parallel(combined_data, num_augmented=2)
        print(f"Augmented data: {augmented_data.shape}")

        # 원본 데이터와 증강된 데이터 결합
        combined_data = pd.concat([combined_data, augmented_data], ignore_index=True)
        print(f"Combined data before deduplication: {combined_data.shape}")

        # 중복된 데이터 제거 (텍스트 기준)
        combined_data = combined_data.drop_duplicates(subset=['text'])
        print(f"Combined data after deduplication: {combined_data.shape}")

    # NaN 값 확인 및 제거
    combined_data = preprocess_data(combined_data)
    print(f"Final data used for training: {combined_data.shape}")

    # 4. 모델 학습 및 평가
    vectorizer, model = train_and_evaluate_model(combined_data)

    # 5. 욕설 판별 및 대체 예제
    texts = ["야 이 나쁜 새끼야", "좋은 아침입니다"]
    for text in texts:
        cleaned_text = replace_swear_words(text, model, vectorizer)
        print(f"Original: {text}")
        print(f"Cleaned: {cleaned_text}")

    # 중복된 데이터를 제거한 combined_data 저장
    combined_data.to_csv(augmented_data_path, sep='|', index=False, header=False)
    print(f"Data saved to {augmented_data_path}")


if __name__ == "__main__":
    main()
