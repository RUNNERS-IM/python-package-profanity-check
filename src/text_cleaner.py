from src.text_preprocessing import preprocess_text
from src.model_training import predict_swearing


def replace_swear_words(text, model, vectorizer):
    # 원본 단어 리스트 생성
    original_words = text.split()
    filtered_words = []

    for original_word in original_words:
        preprocessed_word = preprocess_text(original_word)
        prediction = predict_swearing(preprocessed_word, model, vectorizer)
        print(f"Word: {preprocessed_word}, Original: {original_word}, Prediction: {prediction}")  # 단일 단어 로그 추가
        if prediction == 1:
            filtered_word = original_word.replace(preprocessed_word, '*' * len(preprocessed_word))
            filtered_words.append(filtered_word)
        else:
            filtered_words.append(original_word)

    filtered_text = ' '.join(filtered_words)
    return filtered_text
