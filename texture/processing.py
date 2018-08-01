from enum import Enum
import string
import re


class BreakMode(Enum):
    SPACE = 0
    NO_SPACE = 1
    BEG_OF_TEXT = 2
    END_OF_TEXT = 3


class CapsMode(Enum):
    LOWER = 0
    UPPER = 1
    CAPITALIZED = 2
    MIX_CASE = 3


class Token(object):
    def __init__(self, break_mode, caps_mode, text):
        self.break_mode = break_mode
        self.caps_mode = caps_mode
        self.text = text

    @classmethod
    def parse(cls, text, break_mode=BreakMode.SPACE):
        break_mode = break_mode
        if not text or text.islower():
            caps_mode = CapsMode.LOWER
        elif text.isupper():
            caps_mode = CapsMode.UPPER
        elif text[0].isupper() and text[1:].islower():
            caps_mode = CapsMode.CAPITALIZED
        else:
            caps_mode = CapsMode.MIX_CASE
        text = text if caps_mode == CapsMode.MIX_CASE else text.lower()

        return cls(break_mode, caps_mode, text)

    def __str__(self):
        text = ''
        if self.break_mode == BreakMode.SPACE:
            text += ' '
        text += self.text
        return text

    def __eq__(self, other):
        return (self.text, self.break_mode, self.caps_mode) ==\
                (other.text, other.break_mode, other.caps_mode)


class Document(object):
    PAR_RE = re.compile(r"\. |\.$")

    def __init__(self, language, tokens):
        self.language = language
        self.tokens = tokens

    @classmethod
    def parse(cls, language, text):
        language = language
        tokens = []
        tokens.append(Token.parse("", BreakMode.BEG_OF_TEXT))
        for phrase_text in Document.PAR_RE.split(text):
            tokens.extend(Tokenizer.split(phrase_text + '.'))
        tokens.append(Token.parse("", BreakMode.END_OF_TEXT))
        return cls(language, tokens)

    def __str__(self):
        text = ""
        for token in self.tokens:
            if token.break_mode == BreakMode.BEG_OF_TEXT and text:
                text += ' '
            text += str(token)
        return text


class Tokenizer(object):

    @staticmethod
    def split(text):
        """ tokenize text into tokens

        >>> v = "bought in-!the $10 funny!"
        >>> print([x.text for x in Tokenizer.split(v)])
        ['bought', 'in', '-', '!', 'the', '$', '10', 'funny', '!', '']

        >>> v2 = "  bo in-!the $10 ( fun!"
        >>> print([x.text for x in Tokenizer.split(v2)])
        ['', '', 'bo', 'in', '-', '!', 'the', '$', '10', '(', 'fun', '!', '']
        """
        prev_break = BreakMode.BEG_OF_TEXT
        token = ""
        for ch_i in text:
            if ch_i == ' ':
                if token != "" or prev_break != BreakMode.NO_SPACE:
                    yield Token.parse(token, prev_break)
                    token = ""
                prev_break = BreakMode.SPACE
            elif ch_i in string.punctuation:
                if token != "":
                    yield Token.parse(token, prev_break)
                    prev_break = BreakMode.NO_SPACE
                    token = ""
                token = token + ch_i
                yield Token.parse(token, prev_break)
                prev_break = BreakMode.NO_SPACE
                token = ""
            else:
                token += ch_i

        if token != "":
            yield Token.parse(token, prev_break)

        yield Token.parse("", BreakMode.END_OF_TEXT)

    @staticmethod
    def join(toks_list):
        """ join tokens into text

        >>> v = "bought in-!the $10 funny! !!-cool - #nice"
        >>> print(Tokenizer.join(Tokenizer.split(v)))
        bought in-!the $10 funny! !!-cool - #nice

        >>> v2 = "    bought   in-!the $10 funny! !!-cool - #nice"
        >>> print(Tokenizer.join(Tokenizer.split(v2)))
            bought   in-!the $10 funny! !!-cool - #nice
        """
        text = ""
        for token in toks_list:
            text += str(token)
        return text


if __name__ == "__main__":
    import doctest
    doctest.testmod()
