from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report


def train_and_evaluate_model(data):
    x_train, x_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

    # 데이터 분포 확인
    print(f"Training data distribution: {y_train.value_counts()}")
    print(f"Test data distribution: {y_test.value_counts()}")
    print(f"Training data examples: {x_train.head()}")
    print(f"Test data examples: {x_test.head()}")

    # TF-IDF 벡터화 및 n-그램 범위 조정
    vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    # Naive Bayes 모델 하이퍼파라미터 튜닝
    param_grid = {'alpha': [0.01, 0.1, 0.5, 1.0]}
    grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=5)
    grid_search.fit(x_train_vec, y_train)
    best_nb_model = grid_search.best_estimator_

    # 모델 평가
    y_pred = best_nb_model.predict(x_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Naive Bayes Accuracy: {accuracy * 100:.2f}%')
    print(classification_report(y_test, y_pred))

    # Ensemble 모델
    log_clf = LogisticRegression(max_iter=1000)
    svm_clf = SVC(kernel='linear', probability=True)
    nb_clf = best_nb_model

    voting_clf = VotingClassifier(
        estimators=[('lr', log_clf), ('svc', svm_clf), ('nb', nb_clf)],
        voting='soft'
    )
    voting_clf.fit(x_train_vec, y_train)

    y_pred_ensemble = voting_clf.predict(x_test_vec)
    accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
    print(f'Ensemble Accuracy: {accuracy_ensemble * 100:.2f}%')
    print(classification_report(y_test, y_pred_ensemble))

    scores = cross_val_score(best_nb_model, x_train_vec, y_train, cv=5)
    print(f'Cross-validation accuracy: {scores.mean() * 100:.2f}%')

    return vectorizer, voting_clf


def predict_swearing(text, model, vectorizer):
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)
    return prediction[0]
