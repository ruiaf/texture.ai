import torch

from texture.processing import BreakMode, CapsMode


class DocEmbedding(torch.nn.Module):
    def __init__(self, doc_vocab, hyper_params, sparse=True):
        super(DocEmbedding, self).__init__()

        self.embed_size = hyper_params.universal.embed_size

        self.lang_embed = torch.nn.Embedding(
            len(doc_vocab.lang_vocab),
            hyper_params.doc.lang_embed_size,
            max_norm=1.0,
            sparse=sparse)
        self.break_embed = torch.nn.Embedding(
            len(BreakMode),
            hyper_params.doc.break_embed_size,
            max_norm=1.0,
            sparse=sparse)
        self.caps_embed = torch.nn.Embedding(
            len(CapsMode),
            hyper_params.doc.caps_embed_size,
            max_norm=1.0,
            sparse=sparse)
        self.text_embed = torch.nn.Embedding(
            len(doc_vocab.text_vocab),
            hyper_params.doc.text_embed_size,
            max_norm=1.0,
            padding_idx=0,
            sparse=sparse)

        self.toks_layer = torch.nn.Linear(
            hyper_params.doc.break_embed_size +
            hyper_params.doc.caps_embed_size +
            hyper_params.doc.text_embed_size,
            hyper_params.doc.toks_embed_size)
        self.phrase_dropout = torch.nn.Dropout(p=0.5)
        self.phrase_PReLU = torch.nn.PReLU()

        self.docs_layer = torch.nn.Linear(
            hyper_params.doc.toks_embed_size +
            hyper_params.doc.lang_embed_size,
            self.embed_size)

    def forward(self, lang_inputs, toks_inputs):
        batch_size = toks_inputs.size()[1]

        brks = self.break_embed(toks_inputs[0])
        caps = self.caps_embed(toks_inputs[1])
        text = self.text_embed(toks_inputs[2])

        toks_input = torch.cat((brks, caps, text), dim=2)
        toks_embed = self.toks_layer(toks_input)
        phrase_embed = self.phrase_dropout(
            self.phrase_PReLU(toks_embed.sum(dim=-2)))
        lang_embed = self.lang_embed(lang_inputs.squeeze())

        docs_input = torch.cat((lang_embed, phrase_embed), dim=1)
        docs_embed = self.docs_layer(docs_input)

        return docs_embed

    def decode(self, lang_inputs, toks_inputs):
        return self.forward(lang_inputs, toks_inputs)


class Doc2DocModule(torch.nn.Module):
    def __init__(self, input_encoder, doc_vocab, hyper_params):
        super(Doc2DocModule, self).__init__()

        self.num_neg_samples = hyper_params.doc2doc.num_neg_samples
        self.doc_vocab = doc_vocab
        self.input_encoder = input_encoder
        self.input_transformer = torch.nn.Linear(
            hyper_params.universal.embed_size,
            hyper_params.universal.embed_size)
        self.output_encoder = input_encoder

    def forward(self, lang_inputs, toks_inputs, lang_outputs, toks_outputs):
        batch_size = toks_inputs.size()[1]
        context_size = toks_outputs.size()[2]

        neg_langs = torch.autograd.Variable(
            torch.FloatTensor(
                batch_size * self.num_neg_samples,
                1).uniform_(2, len(self.doc_vocab.lang_vocab) - 1).long(),
            requires_grad=False
        )
        neg_tokens = torch.autograd.Variable(
            torch.stack([
                torch.FloatTensor(
                    batch_size * self.num_neg_samples,
                    context_size).uniform_(0, len(BreakMode) - 1).long(),
                torch.FloatTensor(
                    batch_size * self.num_neg_samples,
                    context_size).uniform_(0, len(CapsMode) - 1).long(),
                torch.FloatTensor(
                    batch_size * self.num_neg_samples,
                    context_size).uniform_(
                        0, len(self.doc_vocab.text_vocab) - 1).long()]),
            requires_grad=False
        )

        input_vecs = self.input_transformer(self.input_encoder(lang_inputs, toks_inputs))
        output_vecs = self.output_encoder(lang_outputs, toks_outputs)
        neg_vectors = self.output_encoder(neg_langs, neg_tokens).view(
            batch_size, -1, self.input_encoder.embed_size).neg()

        oloss = torch.nn.functional.logsigmoid((input_vecs * output_vecs).sum(1).squeeze())
        nloss = torch.nn.functional.logsigmoid(
                torch.bmm(neg_vectors, input_vecs.unsqueeze(2))).sum(1).squeeze()
        return -(oloss + nloss).mean()

    def decode_docs(self, lang_inputs, toks_inputs):
        input_vecs = self.input_encoder(lang_inputs, toks_inputs)
        return input_vecs


class DocClassifierModule(torch.nn.Module):
    def __init__(self, input_encoder, n_classes, hyper_params):
        super(DocClassifierModule, self).__init__()
        self.input_encoder = input_encoder
        transformer_size = int(hyper_params.classification.transformer_ratio * n_classes)
        self.input_transformer = torch.nn.Linear(
            hyper_params.universal.embed_size,
            transformer_size)
        self.transformer_PReLU = torch.nn.PReLU()
        self.output_decoder = torch.nn.Linear(
            transformer_size,
            n_classes)
        self.activation = torch.nn.Softmax(dim=1)
        self.k = min(n_classes, 5)

    def forward(self, lang_inputs, toks_inputs):
        input_vecs = self.input_encoder(lang_inputs, toks_inputs)
        transformed_vecs = self.transformer_PReLU(
                self.input_transformer(input_vecs))
        return self.output_decoder(transformed_vecs)

    def decode_docs(self, lang_inputs, toks_inputs):
        return torch.topk(self.activation(self.forward(lang_inputs, toks_inputs)), self.k)
