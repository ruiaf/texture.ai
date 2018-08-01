import torch
import pyhash
import numpy as np

from texture.processing import BreakMode, CapsMode


class HashTrickVocab(object):
    HASHER = pyhash.xx_32()

    PAD = 0

    def __init__(self, buckets):
        self.buckets = buckets

    def __len__(self):
        return self.buckets

    def encode(self, text):
        return HashTrickVocab.HASHER(text) % self.buckets

    def decode(self, value):
        raise NotImplementedError("HashTickVocab doesn't implement decoding")


class ClassVocab(object):
    def __init__(self):
        self.words = {}
        self.ids = {}
        self.weights = {}

    def __len__(self):
        return len(self.words)

    def gather(self, text, weight):
        if text not in self.ids:
            self.ids[text] = len(self.words)
            self.weights[len(self.words)] = 1.0 / weight
            self.words[len(self.words)] = text

    def encode(self, text):
        return self.ids[text]

    def decode(self, value):
        return self.words[value]

    def class_weight(self, value):
        return self.weights[value]

    def weight_vec(self):
        weight_v = [1.0] * len(self.weights)
        for i, weight in self.weights.items():
            weight_v[i] = weight
        return torch.FloatTensor(weight_v)


class DocumentVocab(object):
    def __init__(self, lang_buckets, text_buckets):
        self.lang_vocab = HashTrickVocab(buckets=lang_buckets)
        self.text_vocab = HashTrickVocab(buckets=text_buckets)


class TokenEncoder(object):
    BREAK_MODES = len(BreakMode)
    CAPS_MODES = len(CapsMode)

    PAD = np.array((
        BreakMode.END_OF_TEXT.value,
        CapsMode.LOWER.value,
        HashTrickVocab.PAD))

    @staticmethod
    def encode(token, text_vocab):
        return np.array((
            token.break_mode.value,
            token.caps_mode.value,
            text_vocab.encode(token.text)))


class DocumentEncoder(object):

    @staticmethod
    def encode(document, doc_vocab, size=20):
        lang = np.array((doc_vocab.lang_vocab.encode(document.language)))
        toks = np.tile(TokenEncoder.PAD, (size, 1))
        for i, token in enumerate(document.tokens):
            if i >= size:
                break
            toks[i] = TokenEncoder.encode(token, doc_vocab.text_vocab)

        return (lang, toks)

    @staticmethod
    def encode_many(doc_list, doc_vocab, max_size=50):
        size = min(max(len(doc.tokens) for doc in doc_list), max_size)

        lang = np.zeros((len(doc_list), 1), np.int64)
        toks = np.zeros((len(doc_list), size, 3), np.int64)
        for i, doc in enumerate(doc_list):
            lang[i], toks[i] = DocumentEncoder.encode(doc, doc_vocab, size)

        return (torch.from_numpy(lang),
                torch.from_numpy(toks.transpose((2, 0, 1))))
