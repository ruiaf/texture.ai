import torch.utils.data

from texture.processing import Document
from texture.encoding import DocumentEncoder, ClassVocab

class Doc2DocDataset(object):

    def __init__(self, tsv_file, doc_vocab):
        self.doc_vocab = doc_vocab
        self.tsv_file = tsv_file

    def batches(self, batch_size=16):
        source_batch = []
        target_batch = []
        while True:
            with open(self.tsv_file) as o_file:
                for i, line in enumerate(o_file):
                    s_doc, t_doc = self.process_line(line)
                    source_batch.append(s_doc)
                    target_batch.append(t_doc)
                    if (i + 1) % batch_size == 0:
                        s_l, s_t = DocumentEncoder.encode_many(source_batch, self.doc_vocab)
                        t_l, t_t = DocumentEncoder.encode_many(target_batch, self.doc_vocab)
                        yield (s_l, s_t, t_l, t_t)
                        source_batch = []
                        target_batch = []

    def process_line(self, line):
        lang, title, content = tuple(line.split('\t'))
        return (Document.parse(lang, title), Document.parse(lang, content))


class DocClassifierDataset(object):

    def __init__(self, classes_file, tsv_file, doc_vocab):
        self.doc_vocab = doc_vocab
        self.tsv_file = tsv_file
        self.class_vocab = ClassVocab()
        with open(classes_file) as o_file:
            for line in o_file:
                class_text, class_pct = line.split('\t')
                self.class_vocab.gather(class_text, float(class_pct))

    def process_line(self, line):
        class_text, content = tuple(line.split('\t'))
        class_id = self.class_vocab.encode(class_text)
        class_weight = self.class_vocab.class_weight(class_id)
        return (Document.parse('<UNK>', content),
                torch.LongTensor([class_id]),
                torch.FloatTensor([class_weight]))

    def batches(self, batch_size=16):
        doc_batch = []
        class_batch = []
        weight_batch = []
        while True:
            with open(self.tsv_file) as o_file:
                for i, line in enumerate(o_file):
                    doc, class_id, weight = self.process_line(line)
                    doc_batch.append(doc)
                    class_batch.append(class_id)
                    weight_batch.append(weight)
                    if (i + 1) % batch_size == 0:
                        lang, toks = DocumentEncoder.encode_many(doc_batch, self.doc_vocab)
                        classes = torch.stack(class_batch)
                        weights = torch.stack(weight_batch)
                        yield (lang, toks, classes, weights)
                        doc_batch = []
                        class_batch = []
                        weight_batch = []
