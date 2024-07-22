import pandas as pd


def load_feedback_data(feedback_file):
    try:
        feedback_data = pd.read_csv(feedback_file, delimiter='|')
    except FileNotFoundError:
        feedback_data = pd.DataFrame(columns=['text', 'label'])
    return feedback_data


def save_feedback(text, is_swearing, feedback_file='data/feedback.txt'):
    feedback_data = pd.DataFrame({'text': [text], 'label': [is_swearing]})
    try:
        existing_feedback = pd.read_csv(feedback_file, delimiter='|')
        feedback_data = pd.concat([existing_feedback, feedback_data], ignore_index=True)
    except FileNotFoundError:
        pass
    feedback_data.to_csv(feedback_file, sep='|', index=False)
