import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pickle
import os
from tensorflow.keras.utils import to_categorical
import tokenization
cls_num = 1
if_albert = False
if_ldl = True
def read_data(filename):
    """
    This function reads the data from .txt file.
    :param filename: reading directory
    :return: lists of word_ids, words, emphasis probabilities, POS tags
    """
    lines = read_lines(filename) + ['']
    word_id_lst, word_id_lsts = [], []
    post_lst, post_lsts = [], []
    bio_lst, bio_lsts = [], []
    freq_lst, freq_lsts = [], []
    e_freq_lst, e_freq_lsts = [], []
    pos_lst, pos_lsts = [], []
    sentences = []
    for line in lines:
        if line:
            splitted = line.split("\t")
            word_id = splitted[0]
            words = splitted[1]
            bio = splitted[2]
            freq = splitted[3]
            e_freq = splitted[4]
            pos = splitted[5]

            word_id_lst.append(word_id)
            post_lst.append(words)
            bio_lst.append(bio)
            freq_lst.append(freq)
            e_freq_lst.append(e_freq)
            pos_lst.append(pos)

        elif post_lst:
            word_id_lsts.append(word_id_lst)
            post_lsts.append(post_lst)
            bio_lsts.append(bio_lst)
            freq_lsts.append(freq_lst)
            e_freq_lsts.append(e_freq_lst)
            pos_lsts.append(pos_lst)
            word_id_lst = []
            post_lst = []
            bio_lst = []
            freq_lst = []
            e_freq_lst = []
            pos_lst = []

    for i in post_lsts:
        sentences.append(" ".join(word for word in i))
    return word_id_lsts, post_lsts, bio_lsts, freq_lsts, e_freq_lsts, pos_lsts, sentences

def read_lines(filename):
    with open(filename, 'r', encoding='utf-8', errors='ignored') as fp:
        lines = [line.strip() for line in fp]
    return lines

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"


# get data list
word_id_lsts, post_lsts, bio_lsts, freq_lsts, e_freq_lsts, pos_lsts, sentences = read_data("./train_dev_data/train.txt")
word_id_lsts1, post_lsts1, bio_lsts1, freq_lsts1, e_freq_lsts1, pos_lsts1, sentences1 = read_data("./train_dev_data/dev.txt")

# get labels and length
if if_ldl:
    train_labels = []
    test_labels = []
    length = []
    for i ,prob_lst in enumerate(e_freq_lsts):
        train_labels.append([])
        length.append(len(prob_lst))
        for j in prob_lst:
            train_labels[i].append([1-float(j),float(j)])
    for i ,prob_lst in enumerate(e_freq_lsts1):
        test_labels.append([])
        length.append(len(prob_lst))
        for j in prob_lst:
            test_labels[i].append([1 - float(j), float(j)])

else:
    train_labels = []
    test_labels = []
    length = []
    train_idxs = []
    for i in e_freq_lsts:
        train_label = []
        length.append(len(i))
        # print(i)
        idx = np.argsort(-np.array([float(k) for k in i]))
        if cls_num == 1:
            idx = idx[:4]
            for j in range(len(i)):
                if j in idx:
                    train_label.append(1)
                else:
                    train_label.append(0)
            train_labels.append(train_label)
        else:
            idx = idx[:cls_num]
            train_idxs.append(idx)
            for j in range(len(i)):
                if j in idx:
                    train_label.append(idx.tolist().index(j))
                else:
                    train_label.append(cls_num - 1)
            train_labels.append(train_label)
    test_idxs = []
    for i in e_freq_lsts1:
        test_label = []
        length.append(len(i))
        idx = np.argsort(-np.array([float(k) for k in i]))
        if cls_num == 1:
            idx = idx[:4]
            for j in range(len(i)):
                if j in idx:
                    test_label.append(1)
                else:
                    test_label.append(0)
            test_labels.append(test_label)
        else:
            idx = idx[:cls_num]
            test_idxs.append(idx)
            for j in range(len(i)):
                if j in idx:
                    test_label.append(idx.tolist().index(j))
                else:
                    test_label.append(cls_num - 1)
            test_labels.append(test_label)
    print(train_labels[0])
    print(test_labels[0])






# get max length
maxlen = max(length)
dict = {}
sum = 0
prob = 1
for i in length:
    if i not in dict.keys():
        dict[i] = 1
    else:
        dict[i] += 1
for i in sorted(dict.items()):
    if sum > prob:
        break
    else:
        sum += i[1] / len(length)
    maxlen = i[0]

# pad labels
if if_ldl:
    for i in train_labels:
        while len(i) < maxlen:
            i.append([0, 0])
    for i in test_labels:
        while len(i) < maxlen:
            i.append([0,0])
else:
    for i in train_labels:
        while len(i) < 38:
            i.append(cls_num - 1)
    for i in test_labels:
        while len(i) < 38:
            i.append(cls_num - 1)

# convert labels to on-hot embedding
print(np.array(train_labels).shape)
print(np.array(test_labels).shape)

if if_ldl:
    val_train_labels = np.array(train_labels)
    val_test_labels = np.array(test_labels)

else:
    val_train_labels = np.reshape(np.array(train_labels),
                                  (np.array(train_labels).shape[0], np.array(train_labels).shape[1], 1))
    val_test_labels = np.reshape(np.array(test_labels),
                                 (np.array(test_labels).shape[0], np.array(test_labels).shape[1], 1))
    if cls_num == 1:
        val_train_labels = to_categorical(val_train_labels, num_classes=cls_num + 1)
        val_test_labels = to_categorical(val_test_labels, num_classes=cls_num + 1)
    else:
        val_train_labels = to_categorical(val_train_labels, num_classes=cls_num)
        val_test_labels = to_categorical(val_test_labels, num_classes=cls_num)

print(val_test_labels)
print(val_train_labels.shape)
print(val_test_labels.shape)


#get_bert_embedding
def convert_sentence_to_features(sentence, tokenizer, max_seq_len):
    tokens = ['[CLS]']
    tokens.extend(tokenizer.tokenize(sentence))
    if len(tokens) > max_seq_len - 1:
        tokens = tokens[:max_seq_len - 1]
    tokens.append('[SEP]')

    segment_ids = [0] * len(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    # Zero Mask till seq_length
    zero_mask = [0] * (max_seq_len - len(tokens))
    input_ids.extend(zero_mask)
    input_mask.extend(zero_mask)
    segment_ids.extend(zero_mask)

    return input_ids, input_mask, segment_ids


def convert_sentences_to_features(sentences, tokenizer, max_seq_len):
    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []

    for sentence in sentences:
        input_ids, input_mask, segment_ids = convert_sentence_to_features(sentence, tokenizer, max_seq_len)
        all_input_ids.append(input_ids)
        all_input_mask.append(input_mask)
        all_segment_ids.append(segment_ids)

    return all_input_ids, all_input_mask, all_segment_ids

# bert_layer = hub.KerasLayer("https://hub.tensorflow.google.cn/tensorflow/bert_en_cased_L-12_H-768_A-12/1",trainable=True)


if if_albert:
    bert_layer = hub.KerasLayer('./albert_base/',trainable=True)
    sp_model_file = bert_layer.resolved_object.sp_model_file.asset_path.numpy()
    tokenizer = tokenization.FullSentencePieceTokenizer(sp_model_file)
else:
    bert_layer = hub.KerasLayer('./bert_cased_768/',trainable=True)
    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
train_vectors = []
test_vectors = []
for i ,sentence in enumerate(sentences):
    input_word_ids, input_mask, segment_ids = convert_sentences_to_features([sentence], tokenizer, maxlen)
    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    if i == 0:
        train_vectors = sequence_output.numpy().tolist()
    else:
        train_vectors.extend(sequence_output.numpy().tolist())

for i ,sentence in enumerate(sentences1):
    input_word_ids, input_mask, segment_ids = convert_sentences_to_features([sentence], tokenizer, maxlen)
    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    if i == 0:
        test_vectors = sequence_output.numpy().tolist()
    else:
        test_vectors.extend(sequence_output.numpy().tolist())

train_vectors = np.array(train_vectors)
test_vectors = np.array(test_vectors)
print(train_vectors.shape)
print(test_vectors.shape)

embedding_dim = train_vectors.shape[2]


#dump word embedding for write results
pickle_file_path = './{}cls_bert_token38_val.pickle'.format(cls_num)
with open(pickle_file_path, 'wb') as pickle_file:
    pickle.dump([train_vectors, test_vectors], pickle_file)

#dump data for running model
model_data_path = './{}cls_bert_embeddding.pickle'.format(cls_num)
with open(model_data_path,'wb') as f:
    pickle.dump([train_vectors,test_vectors,val_train_labels,val_test_labels,train_labels,test_labels,maxlen,embedding_dim],f)
