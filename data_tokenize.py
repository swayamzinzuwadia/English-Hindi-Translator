from tensorflow.keras.preprocessing.text import Tokenizer

def tokenize(data):
    data = data[data["English"].str.len() <= 100]
    print(data.shape)
    data.loc[:, "Hindi"] = data["Hindi"].apply(lambda sentence: "<start> " + sentence + " <end>")

    tok_eng = Tokenizer()
    tok_eng.fit_on_texts(data["English"])

    tok_hindi = Tokenizer()
    tok_hindi.fit_on_texts(data["Hindi"])

    return data, tok_eng, tok_hindi
