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

    # Attention Layer
    attention = MultiHeadAttention(
        num_heads=8, key_dim=32, dropout=0.1, name="attention_layer"
    )(query=decoder_outputs, key=encoder_outputs, value=encoder_outputs)

    # Concatenate attention output with decoder outputs
    decoder_concat_input = Concatenate(axis=-1)([decoder_outputs, attention])

    # Dense Layer for output prediction
    decoder_dense = Dense(output_vocab_size, activation="softmax")
    decoder_outputs = decoder_dense(decoder_concat_input)

    # Full Encoder-Decoder model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Compile the model
    optimizer = AdamW(learning_rate=1e-4, weight_decay=1e-5)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model

def train_model(model, x_train, y_train, x_test, y_test):
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=2, restore_best_weights=True
    )
    model_checkpoint = ModelCheckpoint("model_checkpoint.keras", save_best_only=True)

    history = model.fit(
        x=[x_train, y_train],
        y=y_train,
        batch_size=32,
        epochs=3,
        validation_data=([x_test, y_test], y_test),
        callbacks=[early_stopping, model_checkpoint],
    )

    return history
