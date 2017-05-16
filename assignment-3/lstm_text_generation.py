from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout
from keras.callbacks import ModelCheckpoint

import numpy as np
import random
import pickle
import os


class TextGenerator:
    def __init__(self, seq_len=50, temperature=0.8, epoch=50):
        self.seq_len = seq_len
        self.temperature = temperature
        self.epochs = epoch

    def read_data(self, files_paths):
        self.text = '\n'.join(open(path).read()
                              for path in files_paths).lower()
        self.set_vocab_mapping(self.text)
        return self.text_vec(self.text)

    def text_vec(self, text):
        sentences = []
        next_chars = []
        for i in range(0, len(text) - self.seq_len):
            sentences.append(text[i:i + self.seq_len])
            next_chars.append(text[i + self.seq_len])

        X = np.zeros(
            (len(sentences), self.seq_len, self.vocab_len), dtype=np.bool)
        y = np.zeros((len(sentences), self.vocab_len), dtype=np.bool)
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                X[i, t, self.char_indices[char]] = 1
            y[i, self.char_indices[next_chars[i]]] = 1
        return X, y

    def set_vocab_mapping(self, text):
        chars = sorted(list(set(text)))
        self.vocab_len = len(chars)
        self.char_indices = {c: i for i, c in enumerate(chars)}
        self.indices_char = {i: c for i, c in enumerate(chars)}

    def build_nn(self):
        model = Sequential()
        model.add(LSTM(128, input_shape=(self.seq_len, self.vocab_len)))
        model.add(Dropout(0.2))
        model.add(Dense(self.vocab_len))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        self.model = model

    def fit(self, X, y):
        self.build_nn()
        filepath = "models/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
        checkpoint = ModelCheckpoint(
            filepath,
            monitor='loss',
            verbose=1,
            save_best_only=True,
            mode='min')
        self.model.fit(
            X, y, epochs=self.epochs, batch_size=128, callbacks=[checkpoint])

    def diversity(self, preds):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / self.temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def predict(self, X):
        return self.model.predict(X, verbose=0)[0]

    def generate(self):
        sentence = self.get_random_sentence()
        generated = list()
        for i in range(400):
            X, _ = self.text_vec(sentence)
            next_index = self.diversity(self.predict(X))
            next_char = self.indices_char[next_index]
            sentence = sentence[1:] + next_char
            generated.append(next_char)
        return ''.join(generated)

    def get_random_sentence(self):
        start_index = random.randint(0, len(self.text) - self.seq_len - 1)
        return self.text[start_index:start_index + self.seq_len + 1]

    def load(self, path):
        self.model.load_weights(path)


text_generator = TextGenerator(epoch=1)

X, y = text_generator.read_data('poets/%s' % f for f in os.listdir('poets'))
text_generator.build_nn()
text_generator.fit(X, y)

print(text_generator.generate())
