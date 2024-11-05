import evaluate

def bleu_scoring(hindi_sentences, predicted_sentences):
    blue = evaluate.load("bleu")
    scores = blue.compute(references=hindi_sentences, predictions=predicted_sentences)
    print("BLEU scores: ", scores)
