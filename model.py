from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import (
    LSTM,
    Attention,
    MultiHeadAttention,
    Concatenate,
    Dense,
    Embedding,
    Input,
)
from keras.models import Model
from keras.optimizers import AdamW


def build_model(
    input_vocab_size, output_vocab_size, max_length_input, max_length_output
):
    # Encoder model
    encoder_inputs = Input(shape=(max_length_input,))
    encoder_embedding = Embedding(input_dim=input_vocab_size, output_dim=260)(
        encoder_inputs
    )
    encoder_lstm = LSTM(156, return_sequences=True, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
    encoder_states = [state_h, state_c]

    # Decoder model
    decoder_inputs = Input(shape=(max_length_output,))
    decoder_embedding = Embedding(input_dim=output_vocab_size, output_dim=260)(
        decoder_inputs
    )
    decoder_lstm = LSTM(156, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(
        decoder_embedding, initial_state=encoder_states
    )
