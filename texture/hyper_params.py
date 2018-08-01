# pylint: disable=attribute-defined-outside-init,too-many-instance-attributes

class HyperParams(dict):
    def __init__(self, *args, **kwargs):
        super(HyperParams, self).__init__(*args, **kwargs)
        self.__dict__ = self

    @classmethod
    def default(cls):
        hyper_params = cls()

        universal = cls.default_universal()
        vocab = cls.default_vocab()
        doc = cls.default_doc()
        doc2doc = cls.default_doc2doc()
        classification = cls.default_classification()

        hyper_params.universal = universal
        hyper_params.vocab = vocab
        hyper_params.doc = doc
        hyper_params.doc2doc = doc2doc
        hyper_params.classification = classification

        return hyper_params

    @classmethod
    def default_universal(cls):
        params = cls()

        params.embed_size = 512
        return params

    @classmethod
    def default_vocab(cls):
        params = cls()
        params.lang_buckets = 50
        params.text_buckets = 1000000

        return params

    @classmethod
    def default_doc(cls):
        params = cls()

        params.lang_embed_size = 64
        params.text_embed_size = 512
        params.break_embed_size = 8
        params.caps_embed_size = 8
        params.toks_embed_size = 512
        return params

    @classmethod
    def default_doc2doc(cls):
        params = cls()

        params.num_neg_samples = 5
        return params

    @classmethod
    def default_classification(cls):
        params = cls()
        params.transformer_ratio = 2.0

        return params
