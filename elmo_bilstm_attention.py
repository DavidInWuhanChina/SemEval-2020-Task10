import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from allennlp.modules.elmo import Elmo, batch_to_ids
import numpy as np
# import config
import os
import pickle
import numpy as np
from collections import Counter
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
# import config
from IPython import embed

from flair.data import Sentence
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings
from flair.embeddings import ELMoEmbeddings, BertEmbeddings

from tqdm import tqdm

def read_text_embeddings(filename):
    embeddings = []
    word2index = {}
    with open(filename, 'r', encoding="utf8") as f:
        for i, line in enumerate(f):
            line = line.strip().split()
            word2index[line[0]] = i
            embeddings.append(list(map(float, line[1:])))
    assert len(word2index) == len(embeddings)
    return word2index, np.array(embeddings)

def flatten(elems):
    return [e for elem in elems for e in elem]

class Encoder(object):
    def __init__(self, corpus, emb_path, flair=False):

        self.word2index, self.word_emb = self.get_pretrain_embeddings(emb_path, corpus.get_word_vocab())
        self.index2word = {i: w for w, i in self.word2index.items()}
        self.flair_words = None

        if if_flair or flair:
            # self.elmo = ELMoEmbeddings()
            # self.bert_embedding = BertEmbeddings('bert-base-cased')
            self.flair_forward_embedding = FlairEmbeddings('news-forward')
            self.flair_backward_embedding = FlairEmbeddings('news-backward')
            self.stacked_embeddings = StackedEmbeddings(
                embeddings=[self.flair_forward_embedding, self.flair_backward_embedding])

    def flair_encode(self, data):
        """Generate list of flair embeddings for each sentence in data"""
        sentences = [Sentence(' '.join(words)) for words in data]
        _ = [self.stacked_embeddings.embed(sentence) for sentence in tqdm(sentences)]
        corpus_embeddings = []
        for item in sentences:
            emb_seq = [token.embedding for token in item]
            corpus_embeddings.append(emb_seq)
        return corpus_embeddings

    def encode_words(self, corpus, flair=False):
        if not flair:
            corpus.train.words = [self.encode(self.word2index, sample) for sample in corpus.train.words]
            corpus.dev.words = [self.encode(self.word2index, sample) for sample in corpus.dev.words]
            corpus.test.words = [self.encode(self.word2index, sample) for sample in corpus.test.words]
        else:
            corpus.dev.embeddings = self.flair_encode(corpus.dev.words)
            corpus.train.embeddings = self.flair_encode(corpus.train.words)
            corpus.test.embeddings = self.flair_encode(corpus.test.words)
            return corpus

    def decode_words(self, corpus):
        corpus.train.words = [self.encode(self.index2word, sample) for sample in corpus.train.words]
        corpus.dev.words = [self.encode(self.index2word, sample) for sample in corpus.dev.words]
        corpus.test.words = [self.encode(self.index2word, sample) for sample in corpus.test.words]

    def encode(self, elem2index, elems):
        return [elem2index[elem] for elem in elems]

    @staticmethod
    def get_encoder(corpus, emb_path, encoder_pkl_path):
        if os.path.exists(encoder_pkl_path):
            encoder = Encoder.load(encoder_pkl_path)
        else:
            encoder = Encoder(corpus, emb_path)
            encoder.save(encoder_pkl_path)

        Encoder.print_stats(encoder)

        return encoder

    def print_stats(self):
        print('[LOG]')
        print("[LOG] Word vocab size: {}".format(len(self.word2index)))


    def save(self, filename):
        pickle.dump(self, open(filename, 'wb'))

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as fp:
            return pickle.load(fp)



    def get_pretrain_embeddings(self, filename, vocab):
        assert len(vocab) == len(set(vocab)), "The vocabulary contains repeated words"

        w2i, emb = read_text_embeddings(filename)
        word2index = {'+pad+': 0, '+unk+': 1}
        embeddings = np.zeros((len(vocab) + 2, emb.shape[1]))

        scale = np.sqrt(3.0 / emb.shape[1])
        embeddings[word2index['+unk+']] = np.random.uniform(-scale, scale, (1, emb.shape[1]))

        perfect_match = 0
        case_match = 0
        no_match = 0

        for i in range(len(vocab)):
            word = vocab[i]
            index = len(word2index)  # do not use i because word2index has predefined tokens

            word2index[word] = index
            if word in w2i:
                embeddings[index] = emb[w2i[word]]
                perfect_match += 1
            elif word.lower() in w2i:
                embeddings[index] = emb[w2i[word.lower()]]
                case_match += 1
            else:
                embeddings[index] = np.random.uniform(-scale, scale, (1, emb.shape[1]))
                no_match += 1
        print("[LOG] Word embedding stats -> Perfect match: {}; Case match: {}; No match: {}".format(perfect_match,
                                                                                                     case_match,
                                                                                                     no_match))
        return word2index, embeddings





class Corpus(object):
    def __init__(self, corpus_path):
        self.train = Dataset(os.path.join(corpus_path, 'train.txt'))
        self.dev = Dataset(os.path.join(corpus_path, 'dev.txt'))
        self.test = Dataset1(os.path.join(corpus_path, 'dev.txt'))

    @staticmethod
    def get_corpus(corpus_dir, corpus_pkl_path):
        if os.path.exists(corpus_pkl_path):
            with open(corpus_pkl_path, 'rb') as fp:
                corpus= pickle.load(fp)

        else:
            corpus = Corpus(corpus_dir)
            with open(corpus_pkl_path, 'wb') as fp:
                pickle.dump(corpus, fp, -1)
        corpus.print_stats()
        return corpus

    @staticmethod
    def _get_unique(elems):
        corpus = flatten(elems)
        elems, freqs = zip(*Counter(corpus).most_common())
        return list(elems)


    def print_stats(self):

        print("Train dataset: {}".format(len(self.train.words)))
        print("Dev dataset: {}".format(len(self.dev.words)))
        print("Test dataset: {}".format(len(self.test.words)))

    def get_word_vocab(self):
        return self._get_unique(self.train.words + self.dev.words + self.test.words)
        # return self._get_unique(self.train.words + self.dev.words)
    def get_label_vocab(self):
        return self._get_unique(["O", "I"])

class Dataset(object):
    def __init__(self, path):
        self.words  = self.read_conll_format(path)
        self.labels = self.read_conll_format_labels(path)
        self.embeddings = None

        assert len(self.words) == len(self.labels)


    def read_conll_format_labels(self, filename):
        lines = self.read_lines(filename) + ['']
        posts, post = [], []
        for line in lines:
            if line:
                probs = line.split("\t")[3]
                # reading probabilities from the last column and also normalaize it by div on 9
                probs = [(int(l)/9) for l in probs.split("|")]
                probs = [probs[2],probs[0]+probs[1] ]

                post.append(probs)
                print("post: ", post)
            elif post:
                posts.append(post)
                post = []
        # a list of lists of words/ labels
        return posts

    def read_conll_format(self, filename):
        lines = self.read_lines(filename) + ['']
        posts, post = [], []
        for line in lines:
            if line:
                words = line.split("\t")[1]
                # print("words: ", words)
                post.append(words)
            elif post:
                posts.append(post)
                post = []
        # a list of lists of words/ labels
        return posts

    def read_lines(self, filename):
        with open(filename, 'r') as fp:
            lines = [line.strip() for line in fp]
        return lines
class Dataset1(object):
    def __init__(self, path):
        self.words  = self.read_conll_format(path)
        self.labels = [['I']*len(self.words)]
        self.embeddings = None

        # assert len(self.words) == len(self.labels)

    def read_conll_format(self, filename):
        lines = self.read_lines(filename) + ['']
        posts, post = [], []
        for line in lines:
            if line:
                words = line.split("\t")[1]
                # print("words: ", words)
                post.append(words)
            elif post:
                posts.append(post)
                post = []
        # a list of lists of words/ labels
        return posts

    def read_lines(self, filename):
        with open(filename, 'r') as fp:
            lines = [line.strip() for line in fp]
        return lines
gpu_number = 1

##########################################################
model_mode = "prob"
############################################################
testing = "Flair"
corpus_dir = './train_dev_data/'
output_dir_path = "../models_checkpoints/"+ testing+"/"
# dump_address = "../evals/"+testing+"/"
dump_address = "./"

training = True

if_Elmo = True

if_Bert = False

if_att = True

if_flair = False

if_ROC = True

if_visualize = True

##############################################################
if model_mode== "prob":
    corpus_pkl = corpus_dir + "corpus.io.pkl"
    corpus_pkl_flair = corpus_dir + "corpus.flair.pkl"
    encoder_pkl = corpus_dir + "encoder.io.pkl"
##############################################################
lr = 0.0001
extractor_type = 'lstm'
feat_extractor = 'lstm'

if if_Elmo:
    hidden_dim = 2048
elif if_Bert:
    hidden_dim = 768
elif if_flair:
    hidden_dim = 4096
else:
    hidden_dim = 512

epochs = 2
batch_size = 16

######################################Elmo files##################################################
options_file = "./elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "./elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
########################################################################################
if not torch.cuda.is_available():
    print("[LOG] running on CPU")
    emb_path = './glove.6B.100d.txt'
else:
    print("[LOG] running on GPU")
    emb_path = './glove.6B.100d.txt'


bert_directory = '../../embedding/bert/'


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
# import config
from IPython import embed

class SeqModel(nn.Module):
    def __init__(self, num_labels, extractor_type,  hidden_dim):
        super(SeqModel, self).__init__()
        print("hidden dim: ", hidden_dim)
        # self.wordEmbedding = EmbeddingLayer(embeddings)
        self.featureEncoder = FeatureEncoder(input_dim=4096, extractor_type= extractor_type, hidden_dim =hidden_dim)
        if if_att:
            self.attention = Attention(hidden_dim)

        self.score_layer = nn.Sequential(
            nn.Linear(hidden_dim, 12),
            nn.LayerNorm(12),
            nn.Linear(12, num_labels),
        )



        if torch.cuda.is_available():
            self.wordEmbedding = self.wordEmbedding.cuda()
            self.featureEncoder = self.featureEncoder.cuda()
            if if_att:
                self.attention = self.attention.cuda()
            self.score_layer = self.score_layer.cuda()



    def forward(self, w_tensor, mask):
        # emb_sequence = self.wordEmbedding(w_tensor)  # w_tensor shape: [batch_size, max_seq_len]
        features = self.featureEncoder(w_tensor, mask)  # emb_sequence shape: [batch_size, max_seq_len, emb_dim]

        if if_att:
            att_output, att_weights = self.attention(features, mask.float())
            scores = self.score_layer(att_output) # features shape: [batch_size, max_seq_len, hidden_dim]
        else:
            scores = self.score_layer(features)  # features shape: [batch_size, max_seq_len, hidden_dim]
            att_weights = None
        return scores, att_weights # score shape: [batch_size, max_seq_len, num_labels]



class EmbeddingLayer(nn.Module):
    def __init__(self, embeddings):
        super(EmbeddingLayer, self).__init__()

        self.word_encoder = nn.Sequential(
            nn.Embedding.from_pretrained(torch.from_numpy(embeddings).float(), freeze=False),
            nn.Dropout(0.3)
        )

        if torch.cuda.is_available():
            self.word_encoder = self.word_encoder.cuda()

    def forward(self, w_tensor):
        return self.word_encoder(w_tensor)



class FeatureEncoder(nn.Module):
    def __init__(self, input_dim, extractor_type, hidden_dim):
        super(FeatureEncoder, self).__init__()


        self.extractor_type = extractor_type
        self.hidden_dim = hidden_dim

        if self.extractor_type == 'lstm':
            self.lstm = nn.LSTM(input_dim, self.hidden_dim//2, num_layers=2, batch_first=True, bidirectional=True)
            self.dropout = nn.Dropout(0.4)


            if torch.cuda.is_available():
                self.lstm = self.lstm.cuda()
                self.dropout = self.dropout.cuda()


    def forward(self, sequences, mask):
        """
               :param sequences: sequence shape: [batch_size, seq_len, emb_dim] => [128, 44, 100]
               :param mask:
               :return:
        """
        # embed()
        if self.extractor_type == 'lstm':
            # lengths = torch.sum(mask, 1) # sum up all 1 values which is equal to the lenghts of sequences
            # lengths, order = lengths.sort(0, descending=True)
            # recover = order.sort(0, descending=False)[1]

            # sequences = sequences[order]
            # packed_words = pack_padded_sequence(sequences, lengths.cpu().numpy(), batch_first=True)
            lstm_out, hidden = self.lstm(sequences, None)
            feats = lstm_out
            # feats, _ = pad_packed_sequence(lstm_out)
            # feats = feats.permute(1, 0, 2)
            # feats = feats[recover] # feat shape: [batch_size, seq_len, hidden_dim]
        return feats




class Attention(nn.Module):
    """Attention mechanism written by Gustavo Aguilar https://github.com/gaguilar"""
    def __init__(self,  hidden_size):
        super(Attention, self).__init__()
        self.da = hidden_size
        self.dh = hidden_size

        self.W = nn.Linear(self.dh, self.da)        # (feat_dim, attn_dim)
        self.v = nn.Linear(self.da, 1)              # (attn_dim, 1)

    def forward(self, inputs, mask):
        # Raw scores
        u = self.v(torch.tanh(self.W(inputs)))      # (batch, seq, hidden) -> (batch, seq, attn) -> (batch, seq, 1)

        # Masked softmax
        u = u.exp()                                 # exp to calculate softmax
        u = mask.unsqueeze(2).float() * u           # (batch, seq, 1) * (batch, seq, 1) to zerout out-of-mask numbers
        sums = torch.sum(u, dim=1, keepdim=True)    # now we are sure only in-mask values are in sum
        a = u / sums                                # the probability distribution only goes to in-mask values now

        # Weighted vectors
        z = inputs * a

        return  z,  a.view(inputs.size(0), inputs.size(1))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel
from transformers import BertTokenizer

import numpy as np
# import config

class SeqModel_Bert(nn.Module):
    def __init__(self, num_labels, extractor_type,  hidden_dim):
        super(SeqModel_Bert, self).__init__()

        self.bertLayer = VanillaBertLayer(num_labels)
        # Bert embedding dimension is 768
        self.featureEncoder = FeatureEncoder(input_dim=768, extractor_type= extractor_type, hidden_dim = hidden_dim)
        if if_att:
            self.attention = Attention(hidden_dim)
        self.score_layer = nn.Sequential(
            nn.Linear(hidden_dim, 12),
            nn.LayerNorm(12),
            nn.Linear(12, num_labels),
        )

        if torch.cuda.is_available():
            self.featureEncoder = self.featureEncoder.cuda()
            if if_att:
                self.attention = self.attention.cuda()
            self.score_layer = self.score_layer.cuda()


    def forward(self, tokens):
        emb_sequence, mask = self.bertLayer(tokens)
        features = self.featureEncoder(emb_sequence, mask)  # emb_sequence shape: [batch_size, max_seq_len, emb_dim] => [128, 50, 100]
        if if_att:
            features, att_weights = self.attention(features, mask.float())
        else:
            att_weights = None
        scores = self.score_layer(features) # features shape: [batch_size, max_seq_len, hidden_dim] => [128, 50, 32]
        return scores, mask, att_weights  # score shape: [batch_size, max_seq_len, num_labels] => [128, 50, 3]




class VanillaBertLayer(nn.Module):
    def __init__(self, num_labels):
        super(VanillaBertLayer, self).__init__()
        self.bert = BertModel.from_pretrained(bert_directory, output_hidden_states=True, output_attentions=True, num_labels=num_labels)
        self.tokenizer = BertTokenizer.from_pretrained(bert_directory)


    def forward(self, words):
        # Encode tokens using BertTokenizer
        T = 50
        padded_encodings = []
        attn_masks = []
        segment_ids = []
        for tokens in words:
            padded_tokens = tokens + ['[PAD]' for _ in range(T - len(tokens))]
            attn_mask = [1 if token != '[PAD]' else 0 for token in padded_tokens]
            seg_ids = [0 for _ in range(len(padded_tokens))]
            token_ids = self.tokenizer.encode(padded_tokens)
            padded_encodings.append(token_ids)
            attn_masks.append(attn_mask)
            segment_ids.append(seg_ids)
        token_ids = torch.tensor(padded_encodings)
        attn_mask = torch.tensor(attn_masks)
        seg_ids = torch.tensor(segment_ids)
        hidden_reps, cls_head, hidden_layers,  = self.bert(token_ids, attention_mask = attn_mask, token_type_ids = seg_ids)


        if torch.cuda.is_available():
            hidden_reps = hidden_reps.cuda()
            attn_mask = attn_mask.cuda()
        return hidden_reps, attn_mask


class FeatureEncoder(nn.Module):
    def __init__(self, input_dim, extractor_type, hidden_dim):
        super(FeatureEncoder, self).__init__()


        self.extractor_type = extractor_type
        self.hidden_dim = hidden_dim

        if self.extractor_type == 'lstm':
            self.lstm = nn.LSTM(input_dim, self.hidden_dim//2, num_layers=2, batch_first=True, bidirectional=True)
            self.dropout = nn.Dropout(0.4)

            if torch.cuda.is_available():
                self.lstm = self.lstm.cuda()
                self.dropout = self.dropout.cuda()


    def forward(self, sequences, mask):
        """
       :param sequences: sequence shape: [batch_size, seq_len, emb_dim] => [128, 44, 100]
       :param mask:
       :return:
        """
        if self.extractor_type == 'lstm':
            # lengths = torch.sum(mask, 1) # sum up all 1 values which is equal to the lenghts of sequences
            # lengths, order = lengths.sort(0, descending=True)
            # recover = order.sort(0, descending=False)[1]

            # sequences = sequences[order]
            # packed_words = pack_padded_sequence(sequences, lengths.cpu().numpy(), batch_first=True)
            lstm_out, hidden = self.lstm(sequences, None)
            feats = lstm_out
            # feats, _ = pad_packed_sequence(lstm_out)
            # feats = feats.permute(1, 0, 2)
            # feats = feats[recover] # feat shape: [batch_size, seq_len, hidden_dim] => [128, 44, 32]
        return feats


class Attention(nn.Module):
    """Attention mechanism written by Gustavo Aguilar https://github.com/gaguilar"""
    def __init__(self,  hidden_size):
        super(Attention, self).__init__()
        self.da = hidden_size
        self.dh = hidden_size

        self.W = nn.Linear(self.dh, self.da)        # (feat_dim, attn_dim)
        self.v = nn.Linear(self.da, 1)              # (attn_dim, 1)

    def forward(self, inputs, mask):
        # Raw scores
        u = self.v(torch.tanh(self.W(inputs)))      # (batch, seq, hidden) -> (batch, seq, attn) -> (batch, seq, 1)

        # Masked softmax
        u = u.exp()                                 # exp to calculate softmax
        u = mask.unsqueeze(2).float() * u           # (batch, seq, 1) * (batch, seq, 1) to zerout out-of-mask numbers
        sums = torch.sum(u, dim=1, keepdim=True)    # now we are sure only in-mask values are in sum
        a = u / sums                                # the probability distribution only goes to in-mask values now

        # Weighted vectors
        z = inputs * a

        return  z,  a.view(inputs.size(0), inputs.size(1))
class SeqModel_Elmo(nn.Module):
    def __init__(self, num_labels, extractor_type,  hidden_dim):
        super(SeqModel_Elmo, self).__init__()

        self.elmoLayer = ElmoLayer(options_file, weight_file)
        self.featureEncoder = FeatureEncoder(input_dim=2048, extractor_type= extractor_type, hidden_dim =hidden_dim)
        if if_att:
            self.attention = Attention(hidden_dim)
        self.score_layer = nn.Sequential(
            nn.Linear(hidden_dim, 12),
            nn.LayerNorm(12),
            nn.Linear(12, num_labels),
        )

        if torch.cuda.is_available():
            self.featureEncoder = self.featureEncoder.cuda()
            if if_att:
                self.attention = self.attention.cuda()
            self.score_layer = self.score_layer.cuda()


    def forward(self, words):
        emb_sequence, mask = self.elmoLayer(words)
        features = self.featureEncoder(emb_sequence, mask)  # emb_sequence shape: [batch_size, max_seq_len, emb_dim] => [128, 50, 100]
        if if_att:
            features, att_weights = self.attention(features, mask.float())
        else:
            att_weights = None
        scores = self.score_layer(features) # features shape: [batch_size, max_seq_len, hidden_dim] => [128, 50, 32]
        return scores, mask, att_weights  # score shape: [batch_size, max_seq_len, num_labels] => [128, 50, 3]




class ElmoLayer(nn.Module):
    def __init__(self,options_file, weight_file):
        super(ElmoLayer, self).__init__()
        self.elmo = Elmo(options_file, weight_file, 2, dropout=0.3)



    def forward(self, words):
        character_ids = batch_to_ids(words)
        elmo_output = self.elmo(character_ids)
        elmo_representation = torch.cat(elmo_output['elmo_representations'], -1)
        mask = elmo_output['mask']

        if torch.cuda.is_available():
            elmo_representation = elmo_representation.cuda()
            mask = mask.cuda()
        return elmo_representation, mask


class FeatureEncoder(nn.Module):
    def __init__(self, input_dim, extractor_type, hidden_dim):
        super(FeatureEncoder, self).__init__()


        self.extractor_type = extractor_type
        self.hidden_dim = hidden_dim

        if self.extractor_type == 'lstm':
            self.lstm = nn.LSTM(input_dim, self.hidden_dim//2, num_layers=2, batch_first=True, bidirectional=True)
            self.dropout = nn.Dropout(0.4)

            if torch.cuda.is_available():
                self.lstm = self.lstm.cuda()
                self.dropout = self.dropout.cuda()


    def forward(self, sequences, mask):
        """
       :param sequences: sequence shape: [batch_size, seq_len, emb_dim] => [128, 44, 100]
       :param mask:
       :return:
        """
        if self.extractor_type == 'lstm':
            lengths = torch.sum(mask, 1) # sum up all 1 values which is equal to the lenghts of sequences
            lengths, order = lengths.sort(0, descending=True)
            recover = order.sort(0, descending=False)[1]

            sequences = sequences[order]
            packed_words = pack_padded_sequence(sequences, lengths.cpu().numpy(), batch_first=True)
            lstm_out, hidden = self.lstm(packed_words, None)

            feats, _ = pad_packed_sequence(lstm_out)
            feats = feats.permute(1, 0, 2)
            feats = feats[recover] # feat shape: [batch_size, seq_len, hidden_dim] => [128, 44, 32]
        return feats


class Attention(nn.Module):
    """Attention mechanism written by Gustavo Aguilar https://github.com/gaguilar"""
    def __init__(self,  hidden_size):
        super(Attention, self).__init__()
        self.da = hidden_size
        self.dh = hidden_size

        self.W = nn.Linear(self.dh, self.da)        # (feat_dim, attn_dim)
        self.v = nn.Linear(self.da, 1)              # (attn_dim, 1)

    def forward(self, inputs, mask):
        # Raw scores
        u = self.v(torch.tanh(self.W(inputs)))      # (batch, seq, hidden) -> (batch, seq, attn) -> (batch, seq, 1)

        # Masked softmax
        u = u.exp()                                 # exp to calculate softmax
        u = mask.unsqueeze(2).float() * u           # (batch, seq, 1) * (batch, seq, 1) to zerout out-of-mask numbers
        sums = torch.sum(u, dim=1, keepdim=True)    # now we are sure only in-mask values are in sum
        a = u / sums                                # the probability distribution only goes to in-mask values now

        # Weighted vectors
        z = inputs * a

        return  z,  a.view(inputs.size(0), inputs.size(1))
# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import tensorflow as tf
import numpy as np
import scipy.misc

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO  # Python 3.x


class Logger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

    def log_scalar(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            try:
                s = StringIO()
            except:
                s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)

    def log_histogram(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()
import sys
import time
import torch
import numpy as np

__all__ = ['Helper']


class Helper:
    checkpoint_history = []
    early_stop_monitor_vals = []
    best_score = 0
    best_epoch = 0

    def __init__(self):
        self.USE_GPU = torch.cuda.is_available()

    def checkpoint_model(self, model_to_save, optimizer_to_save, path_to_save, current_score, epoch, mode='min'):
        """
        Checkpoints models state after each epoch.
        :param model_to_save:
        :param optimizer_to_save:
        :param path_to_save:
        :param current_score:
        :param epoch:
        :param n_epoch:
        :param mode:
        :return:
        """
        model_state = {'epoch': epoch + 1,
                       'model_state': model_to_save.state_dict(),
                       'score': current_score,
                       'optimizer': optimizer_to_save.state_dict()}

        # Save the model as a regular checkpoint
        torch.save(model_state, path_to_save + 'last.pth'.format(epoch))

        self.checkpoint_history.append(current_score)
        is_best = False

        # If the model is best so far according to the score, save as the best model state
        if ((np.max(self.checkpoint_history) == current_score and mode == 'max') or
                (np.min(self.checkpoint_history) == current_score and mode == 'min')):
            is_best = True
            self.best_score = current_score
            self.best_epoch = epoch
            # print('inside checkpoint', current_score, np.max(self.checkpoint_history))
            # torch.save(model_state, path_to_save + '{}_best.pth'.format(n_epoch))
            torch.save(model_state, path_to_save + 'best.pth')
            print('BEST saved at epoch: ')
            print("current score: ", current_score)
        if mode=="min":
            print('Current best', round(min(self.checkpoint_history), 4), 'after epoch {}'.format(self.best_epoch))
        else:
            print('Current best', round(max(self.checkpoint_history), 4), 'after epoch {}'.format(self.best_epoch))

        return is_best

    def load_saved_model(self, model, path):
        """
        Load a saved model from dump
        :return:
        """
        # self.active_model.load_state_dict(self.best_model_path)['model_state']
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state'])
        print(">>>>>>>>>>>Loading model form epoch: ", checkpoint['epoch'])
import random, os, numpy, scipy
from codecs import open


# if (k%2 == 0) {
# var heat_text = "<p><br><b>_</b><br>";
# } else {
# var heat_text = "<b>Example:</b><br>";
# }
def createHTML(texts, weights, fileName):
    """
    Creates a html file with text heat.
	weights: attention weights for visualizing
	texts: text on which attention weights are to be visualized
    """
    fileName = "visualization/" + fileName
    fOut = open(fileName, "w", encoding="utf-8")
    part1 = """
    <html lang="en">
    <head>
    <meta http-equiv="content-type" content="text/html; charset=utf-8">
    <style>
    body {
    font-family: Sans-Serif;
    }
    </style>
    </head>
    <body>
    <h3>
    Heatmaps
    </h3>
    </body>
    <script>
    """
    part2 = """
     var color = "255, 70, 50";
    for (var k=0; k < any_text.length; k++) {
        var tokens = any_text[k].split(" ");
        var intensity = new Array(tokens.length);
        var max_intensity = Number.MIN_SAFE_INTEGER;
        var min_intensity = Number.MAX_SAFE_INTEGER;
        for (var i = 0; i < intensity.length; i++) {
            intensity[i] = trigram_weights[k][i];
            if (intensity[i] > max_intensity) {
                max_intensity = intensity[i];
            }
            if (intensity[i] < min_intensity) {
                min_intensity = intensity[i];
            }
        }
    var denominator = max_intensity - min_intensity;
    for (var i = 0; i < intensity.length; i++) {
    intensity[i] = (intensity[i] - min_intensity) / denominator;
    }
    var heat_text = "<p><b>_ </b>";
    var space = "";
    for (var i = 0; i < tokens.length; i++) {
    heat_text += "<span style='background-color:rgba(" + color + "," + intensity[i] + ")'>" + space + tokens[i] + "</span>";
    if (space == "") {
    space = " ";
    }
    }
    //heat_text += "<p>";
    document.body.innerHTML += heat_text;
    }
    </script>
    </html>"""
    putQuote = lambda x: "\"%s\"" % x
    textsString = "var any_text = [%s];\n" % (",".join(map(putQuote, texts)))
    weightsString = "var trigram_weights = [%s];\n" % (",".join(map(str, weights)))
    # print("weightsString:", weightsString)
    fOut.write(part1)
    # print("textsString: ", textsString)
    fOut.write(textsString)
    fOut.write(weightsString)
    fOut.write(part2)
    fOut.close()

    return

from sklearn.metrics import f1_score
import time
import torch
import torch.nn.functional as F
import numpy as np
import random
import sklearn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
# from sklearn.metrics import f1_score
from itertools import chain
from sklearn.metrics import roc_auc_score
# from helper import Helper
# import config
# from logger import Logger
import itertools
# from visualization import attention_visualization
from sklearn_crfsuite import metrics
import pickle
import os
from tqdm import tqdm
from IPython import embed

helper = Helper()
logger = Logger(output_dir_path + 'logs')

def tensor_logging(model, info, epoch):
    for tag, value in info.items():
        logger.log_scalar(tag, value, epoch + 1)
    # Log values and gradients of the model parameters
    for tag, value in model.named_parameters():
        if value.grad is not None:
            tag = tag.replace('.', '/')
            if torch.cuda.is_available():
                logger.log_histogram(tag, value.data.cpu().numpy(), epoch + 1)
                logger.log_histogram(tag + '/grad', value.grad.data.cpu().numpy(), epoch + 1)

def check_predictions(preds, targets, mask):
    overlaped = (preds == targets)
    right = np.sum(overlaped * mask)
    total = mask.sum()
    return right, total, (overlaped * mask)

def visualize_attention(wts,words,filename):
    """
    Visualization function to create heat maps for prediction, ground truth and attention (if any) probabilities
    :param wts:
    :param words:
    :param filename:
    :return:
    """
    wts_add = wts.cpu()
    wts_add_np = wts_add.data.numpy()
    wts_add_list = wts_add_np.tolist()
    text= []
    for index, test in enumerate(words):
        text.append(" ".join(test))
    attention_visualization.createHTML(text, wts_add_list, filename)
    return

def get_batch_all_label_pred(numpy_predictions, numpy_label, mask_numpy, scores_numpy=None):
    """
    To remove paddings
    :param numpy_predictions:
    :param numpy_label:
    :param mask_numpy:
    :param scores_numpy: need this for computing ROC curve
    :return:
    """
    all_label =[]
    all_pred =[]
    all_score = []
    for i in range(len(mask_numpy)):

        all_label.append(list(numpy_label[i][:mask_numpy[i].sum()]))
        all_pred.append(list(numpy_predictions[i][:mask_numpy[i].sum()]))
        if isinstance(scores_numpy, np.ndarray):
            all_score.append(list(scores_numpy[i][:mask_numpy[i].sum()]))

        assert(len(list(numpy_label[i][:mask_numpy[i].sum()]))==len(list(numpy_predictions[i][:mask_numpy[i].sum()])))
        if isinstance(scores_numpy, np.ndarray):
            assert(len(list(numpy_label[i][:mask_numpy[i].sum()])) == len(list(scores_numpy[i][:mask_numpy[i].sum()])))
        assert(len(all_label)==len(all_pred))
    return  (all_label, all_pred) if not isinstance(scores_numpy, np.ndarray) else (all_label, all_pred, all_score)

def to_tensor_labels(encodings,  return_mask=False):
    maxlen = 50 if if_Bert else max(map(len, encodings))
    tensor =[]
    for i, sample in enumerate(encodings):
        seq_len = len(sample)
        padding_len = abs(seq_len - maxlen)
        pad = [[1,0]] * padding_len
        sample.extend(pad)

        tensor.append(sample)
    tensor_tens = torch.Tensor(tensor)

    if torch.cuda.is_available():
        tensor_tens = tensor_tens.cuda()
    return  tensor_tens

def to_tensor(encodings, pad_value=0, return_mask=False):
    maxlen = 50 if if_Bert else max(map(len, encodings))
    tensor = torch.zeros(len(encodings), maxlen).long() + pad_value
    mask = torch.zeros(len(encodings), maxlen).long()
    for i, sample in enumerate(encodings):
        tensor[i, :len(sample)] = torch.tensor(sample, dtype=torch.long)
        mask[i, :len(sample)] = 1
    if torch.cuda.is_available():
        tensor = tensor.cuda()
        mask = mask.cuda()
    return (tensor, mask) if return_mask else tensor

def to_tensor_flair(encodings, pad_value=0, return_mask=False):
    maxlen = 50 if if_Bert else max(map(len, encodings))
    tensor = torch.zeros(len(encodings), maxlen, encodings[0][0].shape[0]).float() + pad_value
    mask = torch.zeros(len(encodings), maxlen).long()
    for i, sample in enumerate(encodings):
        for j, v in enumerate(sample):
            tensor[i,j].add_(sample[j])
        mask[i, :len(sample)] = 1
    if torch.cuda.is_available():
        tensor = tensor.cuda()
        mask = mask.cuda()
    # embed()
    return (tensor, mask) if return_mask else tensor



class Trainer(object):
    def __init__(self, corpus, encoder, batch_size, epochs):
        self.corpus = corpus
        self.encoder = encoder
        self.batch_size = batch_size
        self.batch_size_org = batch_size
        self.epochs = epochs
        self.PAD_TARGET_IX = 0

    def batchify(self, batch_i, dataset, model):
        """
        :param batch_i: ith batch
        :param dataset: train, dev or test set
        :param model: model
        :param theLoss: loss
        :return:
        """

        batch_start = batch_i * self.batch_size_org
        batch_end = batch_start + self.batch_size
        l_tensor = to_tensor_labels(dataset.labels[batch_start: batch_end])
        if if_flair:
            words = dataset.embeddings[batch_start: batch_end]
        else:
            words = dataset.words[batch_start: batch_end]

        if if_Elmo or if_Bert:
            scores, mask, att_w = model.forward(words)
            actual_words_no_pad = words
        elif if_flair:
            w_tensor, mask = to_tensor_flair(words,  return_mask=True)
            # embed()
            scores, att_w = model.forward(w_tensor, mask)
            actual_words_no_pad = dataset.words[batch_start: batch_end]
        else:
            w_tensor, mask= to_tensor(words,  return_mask=True)
            scores, att_w = model.forward(w_tensor, mask) # scores before flatten shape:  [batch_size, seq_len, num_labels]
            w_no_pad = w_tensor.cpu().detach().numpy()

            actual_words_no_pad = [[self.encoder.index2word[elem] for elem in elems] for elems in w_no_pad]

        batch_size, seq_len = l_tensor.size(0), l_tensor.size(1) # target_shape: [batch_size, seq_len]

        scores_flat = F.log_softmax(scores.view(batch_size * seq_len, -1), dim=1) # score_flat shape = [batch_size * seq_len, num_labels]

        target_flat = l_tensor.view(batch_size * seq_len, 2)  # target_flat shape= [batch_size * seq_len]

        return scores, l_tensor, scores_flat, target_flat, seq_len, mask, words,actual_words_no_pad, att_w

    def train(self, model, theLoss, optimizer):
        """
        The train function
        :param model:
        :param theLoss:
        :param optimizer:
        :return:
        """

        print("==========================================================")
        print("[LOG] Training model...")

        total_batch_train = len(self.corpus.train.labels) // self.batch_size
        total_batch_dev = len(self.corpus.dev.labels) // self.batch_size

        if (len(self.corpus.train.labels)) % self.batch_size > 0:
            total_batch_train += 1

        if len(self.corpus.dev.labels) % self.batch_size > 0:
            total_batch_dev += 1


        for epoch in tqdm(range(self.epochs)):
            print("[LOG] Epoch: {epoch+1}/{self.epochs}")
            self.batch_size = self.batch_size_org
            train_total_preds = 0
            train_right_preds = 0
            total_train_loss =0
            model.train()
            train_total_y_true = []
            train_total_y_pred =[]
            with open("output_train.txt", "w") as f:
                for batch_i in tqdm(range(total_batch_train)):

                    if (batch_i == total_batch_train - 1) and (len(self.corpus.train.labels) % self.batch_size > 0):
                        self.batch_size = len(self.corpus.train.labels) % self.batch_size
                    optimizer.zero_grad()
                    score, target, scores_flat, target_flat, seq_len, mask, words,__, _= self.batchify(batch_i, self.corpus.train, model)
                    train_loss = theLoss(scores_flat, F.softmax(target_flat,dim=1))#/ self.batch_size
                    target_flat_softmaxed = F.softmax(target_flat, 1)



                    train_loss.backward()
                    optimizer.step()
                    total_train_loss += train_loss.item() * self.batch_size


                    _, predictions_max = torch.max(torch.exp(scores_flat), 1)
                    predictions_max = predictions_max.view(self.batch_size, seq_len)
                    numpy_predictions_max = predictions_max.cpu().detach().numpy()


                    _, label_max = torch.max(target_flat_softmaxed, 1)
                    label_max = label_max.view(self.batch_size, seq_len)
                    numpy_label_max = label_max.cpu().detach().numpy()


                    #mask:
                    mask_numpy = mask.cpu().detach().numpy()
                    right, whole, overlaped = check_predictions(numpy_predictions_max, numpy_label_max, mask_numpy)
                    train_total_preds += whole
                    train_right_preds += right
                    all_label, all_pred = get_batch_all_label_pred(numpy_predictions_max, numpy_label_max, mask_numpy)
                    train_total_y_pred.extend(all_pred)
                    train_total_y_true.extend(all_label)




            train_f1_total = metrics.flat_f1_score(train_total_y_true, train_total_y_pred, average= "micro")

            train_loss = total_train_loss/ len(self.corpus.train.labels)
            print("[lOG] ++Train_loss: {}++, ++MAX train_accuracy: {}++, ++MAX train_f1_score: {}++ ".format(train_loss, (train_right_preds / train_total_preds), (train_f1_total) ))




            print("[LOG] ______compute dev: ")
            model.eval()
            self.batch_size = self.batch_size_org

            dev_right_preds = 0
            dev_total_preds = 0
            total_dev_loss = 0
            dev_total_y_true = []
            dev_total_y_pred = []
            for batch_i in range(total_batch_dev):
                if (batch_i == total_batch_dev - 1) and (len(self.corpus.dev.labels) % self.batch_size > 0):
                    self.batch_size = len(self.corpus.dev.labels) % self.batch_size

                dev_score, dev_target,dev_scores_flat, dev_target_flat, dev_seq_len, dev_mask, dev_words,__, _= self.batchify(batch_i, self.corpus.dev, model)
                dev_loss = theLoss(dev_scores_flat, F.softmax(dev_target_flat, 1)) #/ self.batch_size
                total_dev_loss += dev_loss.item() * self.batch_size
                dev_target_flat_softmaxed = F.softmax(dev_target_flat, 1)

                _, dev_predictions_max = torch.max(dev_scores_flat, 1)
                dev_predictions_max = dev_predictions_max.view(self.batch_size, dev_seq_len)
                dev_numpy_predictions_max = dev_predictions_max.cpu().detach().numpy()


                _, dev_label_max = torch.max(dev_target_flat_softmaxed, 1)
                dev_label_max = dev_label_max.view(self.batch_size, dev_seq_len)
                dev_numpy_label_max = dev_label_max.cpu().detach().numpy()


                # mask:
                dev_mask_numpy = dev_mask.cpu().detach().numpy()

                dev_right, dev_whole, dev_overlaped = check_predictions(dev_numpy_predictions_max, dev_numpy_label_max, dev_mask_numpy)
                dev_total_preds += dev_whole
                dev_right_preds += dev_right

                all_label, all_pred = get_batch_all_label_pred(dev_numpy_predictions_max, dev_numpy_label_max, dev_mask_numpy, 0)
                dev_total_y_pred.extend(all_pred)
                dev_total_y_true.extend(all_label)


            else:
                dev_f1_total_micro = metrics.flat_f1_score(dev_total_y_true, dev_total_y_pred, average= "micro")
            dev_loss = total_dev_loss / len(self.corpus.dev.labels)
            dev_f1_total_macro = metrics.flat_f1_score(dev_total_y_true, dev_total_y_pred, average="macro")

            #checkpoint:
            is_best = helper.checkpoint_model(model, optimizer, output_dir_path, dev_loss, epoch + 1, 'min')

            print("<<dev_loss: {}>> <<dev_accuracy: {}>> <<dev_f1: {}>> ".format( dev_loss, (dev_right_preds / dev_total_preds), (dev_f1_total_micro)))
            print("--------------------------------------------------------------------------------------------------------------------------------------------------")
            #tensorBoard:
            info = {'training_loss': train_loss,
                    'train_accuracy': (train_right_preds / train_total_preds),
                    'train_f1': (train_f1_total),
                    'validation_loss': dev_loss,
                    'validation_accuracy': (dev_right_preds / dev_total_preds),
                    'validation_f1_micro': (dev_f1_total_micro),
                    'validation_f1_macro': (dev_f1_total_macro)
                    }
            tensor_logging(model, info, epoch)

    def predict(self, model, theLoss, theCorpus,dump_adress):
        print("==========================================================")
        print("Predicting...")
        helper.load_saved_model(model, output_dir_path + 'best.pth')
        model.eval()
        self.batch_size = self.batch_size_org
        total_batch_test = len(theCorpus.labels) // self.batch_size
        if len(theCorpus.words) % self.batch_size > 0:
            total_batch_test += 1

        test_right_preds, test_total_preds = 0, 0
        test_total_y_true = []
        test_total_y_pred = []
        test_total_y_scores = []
        total_scores_numpy_probs =[]
        total_labels_numpy_probs =[]
        total_mask_numpy =[]
        total_test_loss = 0
        with open("output_test.txt", "w") as f:
            for batch_i in range(total_batch_test):
                if (batch_i == total_batch_test - 1) and (len(theCorpus.words) % self.batch_size > 0):
                    self.batch_size = len(theCorpus.words) % self.batch_size
                score, target, scores_flat, target_flat, seq_len, mask, words,actual_words_no_pad,  att_w = self.batchify(batch_i, theCorpus, model)
                test_loss =  theLoss(scores_flat, F.softmax(target_flat, 1)) #/ self.batch_size

                total_test_loss += test_loss.item() * self.batch_size
                scores_flat_exp = torch.exp(scores_flat)

                print("--[LOG]-- test loss: ", test_loss)

                _, predictions_max = torch.max(scores_flat_exp, 1)

                predictions_max = predictions_max.view(self.batch_size, seq_len)
                numpy_predictions_max = predictions_max.cpu().detach().numpy()

                # computing scores for ROC curve:
                scores_numpy = scores_flat_exp[:, 1].view(self.batch_size, seq_len)
                scores_numpy = scores_numpy.cpu().detach().numpy()

                total_scores_numpy_probs.extend(scores_numpy)

                # if based on MAX
                _, label_max = torch.max(target_flat, 1)
                label_max = label_max.view(self.batch_size, seq_len)
                numpy_label_max = label_max.cpu().detach().numpy()
                # for computing senetnce leveL:
                total_labels_numpy_probs.extend(target_flat[:, 1].view(self.batch_size, seq_len).cpu().detach().numpy())

                # mask:
                mask_numpy = mask.cpu().detach().numpy()
                total_mask_numpy.extend(mask_numpy)
                right, whole, overlaped = check_predictions(numpy_predictions_max, numpy_label_max, mask_numpy)
                test_total_preds += whole
                test_right_preds += right
                all_label, all_pred, all_scores= get_batch_all_label_pred(numpy_predictions_max, numpy_label_max, mask_numpy, scores_numpy)
                test_total_y_pred.extend(all_pred)
                test_total_y_true.extend(all_label)

                #ROC:
                if if_ROC:
                    test_total_y_scores.extend(all_scores)

                # Visualization:
                if if_visualize:
                    sfe = scores_flat_exp[:, 1].view(self.batch_size, seq_len)
                    visualize_attention(sfe, actual_words_no_pad, filename='res/scores'+str(batch_i)+'.html')
                    visualize_attention(target[:,:,1], actual_words_no_pad, filename='res/target' + str(batch_i) + '.html')
                    visualize_attention(F.softmax(target, 1)[:,:,1], actual_words_no_pad, filename='res/target_softmaxed' + str(batch_i) + '.html')


            test_f1_total_micro = metrics.flat_f1_score(test_total_y_true, test_total_y_pred, average= "micro")
            test_f1_total_macro = metrics.flat_f1_score(test_total_y_true, test_total_y_pred, average="macro")
            test_f1_total_binary = metrics.flat_f1_score(test_total_y_true, test_total_y_pred, average="binary")

            roc_score= roc_auc_score(list(itertools.chain(*test_total_y_true)) , list(itertools.chain(*test_total_y_scores)))
            test_loss = total_test_loss / len(self.corpus.test.labels)
            pickle.dump(list(itertools.chain(*test_total_y_true)),
                open(os.path.join(dump_address, "y_true.pkl"), "wb"))
            pickle.dump(list(itertools.chain(*test_total_y_scores)),
                        open(os.path.join(dump_address, "y_pred.pkl"), "wb"))

            print(
                "->>>>>>>>>>>>>TOTAL>>>>>>>>>>>>>>>>>>>>>>> test_loss: {}, test_accuracy: {}, test_f1_score_micro: {} ROC:{}".format(
                    test_loss, (test_right_preds / test_total_preds), (test_f1_total_micro), roc_score))
            print()
            print(metrics.flat_classification_report(test_total_y_true, test_total_y_pred))
            print("test_f1_total_binary: ", test_f1_total_binary)
            print("precision binary: ", metrics.flat_precision_score(test_total_y_true, test_total_y_pred, average="binary"))
            print("recall binary: ", metrics.flat_recall_score(test_total_y_true, test_total_y_pred, average="binary"))


            if not os.path.exists(dump_address):
                os.makedirs(dump_address)
            print("[LOG] dumping results in ", dump_address)
            pickle.dump(np.array(total_scores_numpy_probs),
                        open(os.path.join(dump_address, "score_pobs.pkl"), "wb"))
            pickle.dump(np.array(total_labels_numpy_probs),
                        open(os.path.join(dump_address, "label_pobs.pkl"), "wb"))
            pickle.dump(np.array(total_mask_numpy), open(os.path.join(dump_address, "mask_pobs.pkl"), "wb"))

import os
import argparse
import torch
import json
import re
import torch
import torch.nn as nn
import pickle
import numpy as np
from torch.optim import SGD
# from utils.data  import Corpus, Encoder
# from model.seqmodel import SeqModel
# from model.seqmodel_Elmo import SeqModel_Elmo
# from model.seqmodel_Bert import SeqModel_Bert
#from model.lstm_crf import Lstm_crf
import torch.optim as optim
# import config
# from config import *
from IPython import embed

if __name__ == '__main__':
    if torch.cuda.is_available():
        print("Running on GPU {}".format(gpu_number))
        torch.cuda.set_device(gpu_number)
    else:
        print("Running on CPU")

    print("[LOG] dumping in .. ", dump_address)
    if not training:
        print("[LOG] NO training ...!")



    torch.manual_seed(0)
    np.random.seed(0)



    corpus = Corpus.get_corpus(corpus_dir, corpus_pkl)

    if if_flair:
        # encoder = Encoder(corpus, emb_path, flair=True)
        # with open(corpus_pkl_flair, 'wb') as fp:
        #     pickle.dump(corpus, fp, -1)

        with open(corpus_pkl_flair, 'rb') as fp:
            corpus = pickle.load(fp)
        encoder = None
    else:
        encoder = Encoder.get_encoder(corpus, emb_path, encoder_pkl)

        if not (if_Elmo or if_Bert):
            encoder.encode_words(corpus, flair=True)

        embed()

    if model_mode=="prob":
        # from trainer_prob import Trainer
        theLoss = nn.KLDivLoss(reduction='elementwise_mean')#size_average=True)
        if if_Elmo:
            torch.backends.cudnn.enabled = False
            print("[LOG] Using Elmo ...")
            model = SeqModel_Elmo(len(corpus.get_label_vocab()), extractor_type,  hidden_dim)
        elif if_Bert:
            print("[LOG] Using Bert ...")
            model = SeqModel_Bert(len(corpus.get_label_vocab()), extractor_type,  hidden_dim)
        else:
            if if_flair:
                print("[LOG] Using Flair ...")
            model = SeqModel(len(corpus.get_label_vocab()), extractor_type,  hidden_dim)
        optimizer = optim.Adam(lr=lr, params=model.parameters())
        print("==========================================================")
        print("[LOG] Model:")
        print(model)
        print("==========================================================")
        print("[LOG] Train:")
        trainer = Trainer(corpus, encoder, batch_size, epochs)
        if training:
            trainer.train(model, theLoss, optimizer)


        print("==========================================================")
        print("[LOG] Test:")

        trainer.predict(model, theLoss, trainer.corpus.test,dump_address)
        embed()