import numpy as np
import pandas as pd
import random
import ray


def augment_data_parallel(data, num_augmented=1):
    ray.init(ignore_reinit_error=True)
    data_chunks = np.array_split(data, 4)
    augmented_chunks = ray.get([augment_chunk.remote(chunk, num_augmented) for chunk in data_chunks])
    augmented_data = pd.concat(augmented_chunks, ignore_index=True)
    return augmented_data


@ray.remote
def augment_chunk(chunk, num_augmented):
    augmented_texts = chunk['text'].apply(lambda x: synonym_replacement(x, num_augmented))
    augmented_labels = chunk['label']
    augmented_df = pd.DataFrame({'text': augmented_texts, 'label': augmented_labels})
    return augmented_df


def synonym_replacement(text, num_augmented):
    words = text.split()
    new_words = words.copy()
    random_word_list = list(set(words))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(synonyms)
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= num_augmented:
            break
    sentence = ' '.join(new_words)
    return sentence


def get_synonyms(word):
    # 간단한 한국어 동의어 사전 정의
    synonym_dict = {
        '행복하다': ['기쁘다', '즐겁다', '유쾌하다'],
        '슬프다': ['우울하다', '비통하다', '애처롭다'],
        '좋다': ['훌륭하다', '나이스하다', '멋지다'],
        '나쁘다': ['불량하다', '형편없다', '못되다'],
        '빠르다': ['신속하다', '재빠르다', '민첩하다'],
        '느리다': ['천천히', '늦다', '지체하다'],
        '힘들다': ['어렵다', '고생하다', '힘겹다'],
        '쉽다': ['간단하다', '용이하다', '수월하다'],
        '비싸다': ['고가다', '값비싸다', '비용이많다'],
        '싸다': ['저렴하다', '값싸다', '저가다'],
        '좋아하다': ['사랑하다', '선호하다', '즐기다'],
        '싫어하다': ['미워하다', '꺼리다', '혐오하다'],
        '크다': ['거대하다', '거창하다', '큰'],
        '작다': ['작은', '소형', '미량의'],
    }
    return synonym_dict.get(word, [])
