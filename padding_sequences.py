from tensorflow.keras.preprocessing.sequence import pad_sequences

def padding(data, tok_eng, tok_hindi):
    data["English"] = tok_eng.texts_to_sequences(data["English"])
    data["Hindi"] = tok_hindi.texts_to_sequences(data["Hindi"])
    
    max_length_combined = max(
        max(len(seq) for seq in data["English"]), 
        max(len(seq) for seq in data["Hindi"])
    )
    
    x = pad_sequences(data["English"], maxlen=max_length_combined, padding="post")
    y = pad_sequences(data["Hindi"], maxlen=max_length_combined, padding="post")
    
    return x, y