from collections import deque
import numpy as np
import time
import torch
import torch.multiprocessing as mp
import os.path

from texture.encoding import DocumentVocab, DocumentEncoder
from texture.models import DocClassifierModule, Doc2DocModule, DocEmbedding
from texture.hyper_params import HyperParams

class SingleTask(mp.Process):
    def __init__(self, scheduler, task_name, train_set, test_set):
        super(SingleTask, self).__init__()
        self.scheduler = scheduler
        self.train_set = train_set
        self.test_batches = test_set.batches()
        self.train_batches = train_set.batches()
        self.train_losses = deque()
        self.test_losses = deque()
        self.iteration = 0
        self.task_name = task_name
        self.optimizer = None
        self.model = None

    def run(self):
        for _ in range(1000000000):
            self.scheduler.put(None)
            for _ in range(100):
                self.iterate()
            print(self.task_name, self.evaluate())
            self.scheduler.get()

    def iterate(self):
        self.iteration += 1
        self.model.train()
        train_loss = self.compute(*next(self.train_batches))
        self.optimizer.zero_grad()
        train_loss.backward()
        self.optimizer.step()

        test_loss = self.compute(*next(self.test_batches))

        self.record_loss(train_loss.data[0], test_loss.data[0])

    def record_loss(self, train_loss, test_loss):
        self.train_losses.appendleft(train_loss)
        self.test_losses.appendleft(test_loss)
        if len(self.train_losses) > 1000:
            self.train_losses.pop()
            self.test_losses.pop()

    def evaluate(self):
        return (self.iteration,
                np.mean(self.train_losses),
                np.mean(self.test_losses))

    def compute(self):
        raise NotImplementedError("Abstract class doesn't implement run")



class Doc2DocTask(SingleTask):
    def __init__(self, scheduler, task_name, train_set, test_set, input_encoder, hyper_params):
        super(Doc2DocTask, self).__init__(scheduler, task_name, train_set, test_set)
        self.model = Doc2DocModule(input_encoder, train_set.doc_vocab, hyper_params)
        self.model.share_memory()
        print(self.model)
        self.learning_rate = 0.0000001
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate)

    def compute(self, src_langs, src_toks, target_langs, target_toks):
        lang_ivec = torch.autograd.Variable(
            src_langs,
            requires_grad=False)
        toks_ivec = torch.autograd.Variable(
            src_toks,
            requires_grad=False)
        langs_ovec = torch.autograd.Variable(
            target_langs,
            requires_grad=False)
        toks_ovec = torch.autograd.Variable(
            target_toks,
            requires_grad=False)

        loss = self.model(lang_ivec, toks_ivec, langs_ovec, toks_ovec)
        return loss

    def decode_docs(self, langs, tokens):
        self.model.eval()
        langs = torch.autograd.Variable(langs, requires_grad=False)
        toks = torch.autograd.Variable(tokens, requires_grad=False)
        return self.model.decode_docs(langs, toks)


class DocClassifierTask(SingleTask):
    def __init__(self, scheduler, task_name, train_set, test_set, input_encoder, hyper_params):
        super(DocClassifierTask, self).__init__(scheduler, task_name, train_set, test_set)
        self.n_classes = len(train_set.class_vocab)
        self.loss = torch.nn.CrossEntropyLoss(
                train_set.class_vocab.weight_vec())

        self.model = DocClassifierModule(
            input_encoder,
            self.n_classes,
            hyper_params)
        self.model.share_memory()
        print(self.model)
        self.learning_rate = 0.01
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.learning_rate)

    def compute(self, langs, toks, labels, weights):
        lang_ivec = torch.autograd.Variable(
            langs,
            requires_grad=False)
        toks_ivec = torch.autograd.Variable(
            toks,
            requires_grad=False)
        true_ovecs = torch.autograd.Variable(
            labels,
            requires_grad=False).squeeze()

        pred_ovecs = self.model(lang_ivec, toks_ivec).squeeze()
        loss = self.loss(pred_ovecs, true_ovecs)
        return loss

    def decode_docs(self, langs, toks):
        self.model.eval()
        lang_vecs = torch.autograd.Variable(langs, requires_grad=False)
        toks_vecs = torch.autograd.Variable(toks, requires_grad=False)
        probs, classes = self.model.decode_docs(lang_vecs, toks_vecs)

        results = []
        for i in range(probs.size()[0]):
            results.append([])
            for j in range(probs.size()[1]):
                results[i].append(
                    (self.train_set.class_vocab.decode(
                        classes[i][j].data[0]),
                     probs[i][j].data[0]))
        return results

class MultiTaskModel(object):
    def __init__(self):
        self.hyper_params = HyperParams.default()
        self.doc_vocab = DocumentVocab(
            self.hyper_params.vocab.lang_buckets,
            self.hyper_params.vocab.text_buckets)
        self.doc_encoder = DocEmbedding(self.doc_vocab, self.hyper_params)
        self.doc_encoder.share_memory()
        self.scheduler = mp.Queue(mp.cpu_count() - 1)
        self.tasks = {}

    def add_doc2doc_task(self, task_name, train_set, test_set):
        task = Doc2DocTask(self.scheduler, task_name, train_set, test_set, self.doc_encoder, self.hyper_params)
        self.tasks[task_name] = task

    def add_doc_classification_task(self, task_name, train_set, test_set):
        task = DocClassifierTask(self.scheduler, task_name, train_set, test_set, self.doc_encoder, self.hyper_params)
        self.tasks[task_name] = task

    def train(self, wait=True):
        for _, task in self.tasks.items():
            task.start()

        if wait:
            for _, task in self.tasks.items():
                task.join()

    def decode_docs(self, docs):
        langs, toks = DocumentEncoder.encode_many(docs, self.doc_vocab)

        results = {}
        for task_name, task in self.tasks.items():
            if hasattr(task.__class__, 'decode_docs'):
                results[task_name] = task.decode_docs(langs, toks)
        return results

    def load_model(self, folder):
        for task_name, task in self.tasks.items():
            m_path = folder + "/" + task_name + ".mdl"
            if os.path.isfile(m_path):
                print("Loading %s" % task_name)
                model_dict = task.model.state_dict()
                model_dict.update(torch.load(m_path))
                task.model.load_state_dict(model_dict)

    def save_model(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
        for task_name, task in self.tasks.items():
            m_path = folder + "/" + task_name + ".mdl"
            print("Saving %s" % task_name)
            torch.save(task.model.state_dict(), m_path)
