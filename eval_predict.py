import numpy as np
import evaluate
from rich import print
from IPython.display import display


def bleu_scoring(hindi_sentences, predicted_sentences):
    blue = evaluate.load("bleu")
    scores = blue.compute(references=hindi_sentences, predictions=predicted_sentences)
    print("BLEU scores: ", scores)


def get_rev_token_dict(tokenizer):
    return {idx: word for word, idx in tokenizer.word_index.items()}


def select_test_subset(x_test, y_test, num_samples=101):
    x_test_subset = x_test[1:num_samples]
    y_test_padded_subset = y_test[1:num_samples]
    return x_test_subset, y_test_padded_subset


def predict_translations(model, x_test_subset, y_test_padded_subset, batch_size=16):
    predictions = model.predict(
        [x_test_subset, y_test_padded_subset], batch_size=batch_size
    )
    print("Shape of predictions:", predictions.shape)
    return predictions


def convert_predictions_to_tokens(predictions):
    predicted_tokens_np = np.argmax(predictions, axis=-1)
    print("Shape of predicted_tokens:", predicted_tokens_np.shape)
    return predicted_tokens_np


def map_indices_to_tokens(predicted_tokens_np, rev_tok_hindi, tok_hindi):
    predicted_sentences = []
    for sample in predicted_tokens_np:
        sentence = " ".join(
            [
                rev_tok_hindi.get(token, "<unknown>")
                for token in sample
                if token != 0
                and token
                not in [
                    tok_hindi.word_index.get("start"),
                    tok_hindi.word_index.get("end"),
                ]
            ]
        )
        predicted_sentences.append(sentence)
    return predicted_sentences


def map_sentences(x_test_subset, rev_tok):
    sentences = []
    for sample in x_test_subset:
        sentence = " ".join(
            [rev_tok.get(token, "<unknown>") for token in sample if token != 0]
        )
        sentences.append(sentence)
    return sentences


def print_translations(english_sentences, predicted_sentences):
    for idx, (eng_sentence, hin_sentence) in enumerate(
        zip(english_sentences, predicted_sentences)
    ):
        display(f"English sentence {idx + 1}: {eng_sentence}")
        display(f"Predicted Hindi translation {idx + 1}: {hin_sentence}")
