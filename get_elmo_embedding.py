import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pickle
import os
from tensorflow.keras.utils import to_categorical
cls_num = 5

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

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# val_train_labels, train_labels, val_test_labels, test_labels, label_tag_dict, tag_label_dict = pickle.load(open(pickle_file, 'rb'))
# get data list
word_id_lsts, post_lsts, bio_lsts, freq_lsts, e_freq_lsts, pos_lsts, sentences = read_data("./train_dev_data/train.txt")
word_id_lsts1, post_lsts1, bio_lsts1, freq_lsts1, e_freq_lsts1, pos_lsts1, sentences1 = read_data("./train_dev_data/dev.txt")


#get labels and length
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
                train_label.append(cls_num-1)
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
                test_label.append(cls_num-1)
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
    
#pad labels
for i in train_labels:
    while len(i) < 38:
        i.append(cls_num-1)
for i in test_labels:
    while len(i) < 38:
        i.append(cls_num-1)

#convert labels to on-hot embedding
print(np.array(train_labels).shape)
print(np.array(test_labels).shape)

val_train_labels = np.reshape(np.array(train_labels),(np.array(train_labels).shape[0], np.array(train_labels).shape[1], 1))
if cls_num == 1:
    val_train_labels = to_categorical(val_train_labels, num_classes=cls_num+1)
else:
    val_train_labels = to_categorical(val_train_labels, num_classes=cls_num)

val_test_labels = np.reshape(np.array(test_labels), (np.array(test_labels).shape[0], np.array(test_labels).shape[1], 1))
if cls_num == 1:
    val_test_labels = to_categorical(val_test_labels, num_classes=cls_num+1)
else:
    val_test_labels = to_categorical(val_test_labels, num_classes=cls_num)
print(val_test_labels)
print(val_train_labels.shape)
print(val_test_labels.shape)


#get elmo word embedding ,padding and convert tensor to numpy
elmo = hub.KerasLayer("https://hub.tensorflow.google.cn/google/elmo/3", trainable=True, output_key='elmo')
train_vectors = []
test_vectors = []
for i in range(len(sentences)):
    temp = elmo(tf.convert_to_tensor([sentences[i]]))
    temp = temp.numpy()
    temp = temp.tolist()
    for j in temp:
        if len(j) < maxlen:
            j.extend([[0.0] * 1024] * (maxlen - len(j)))
    if i == 0:
        train_vectors = temp
    else:
        train_vectors.extend(temp)
for i in range(len(sentences1)):
    temp = elmo(tf.convert_to_tensor([sentences1[i]]))
    temp = temp.numpy()
    temp = temp.tolist()
    for j in temp:
        if len(j) < maxlen:
            j.extend([[0.0] * 1024] * (maxlen - len(j)))
    if i == 0:
        test_vectors = temp
    else:
        test_vectors.extend(temp)

train_vectors = np.array(train_vectors)
test_vectors = np.array(test_vectors)
print(train_vectors.shape)
print(test_vectors.shape)

embedding_dim = train_vectors.shape[2]

# re_train_idxs = []
# re_test_idxs = []
#
# for i in range(train_vectors.shape[0]):
#     re_train_idxs.append([])
#     for j in range(train_vectors.shape[1]):
#         if j not in train_idxs[i]:
#             re_train_idxs[i].append(j)
# for i in range(test_vectors.shape[0]):
#     re_test_idxs.append([])
#     for j in range(test_vectors.shape[1]):
#         if j not in test_idxs[i]:
#             re_test_idxs[i].append(j)
# print(re_train_idxs[0])



#
# if cls_num == 4:
#     val_train_vectors = []
#     val_test_vectors = []
#     val_train_labels1 = []
#     val_test_labels1 = []
#     train_labels1 = []
#     test_labels1 = []
#     for i in range(train_vectors.shape[0]):
#             val_train_vectors.append(np.delete(train_vectors[i],re_train_idxs[i],axis=0))
#             val_train_labels1.append(np.delete(val_train_labels[i], re_train_idxs[i], axis=0))
#             train_labels1.append(np.delete(np.array(train_labels)[i], re_train_idxs[i], axis=0))
#     for i in range(test_vectors.shape[0]):
#             val_test_vectors.append(np.delete(test_vectors[i],re_test_idxs[i],axis=0))
#             val_test_labels1.append(np.delete(val_test_labels[i], re_test_idxs[i], axis=0))
#             test_labels1.append(np.delete(np.array(test_labels)[i], re_test_idxs[i], axis=0))
#     maxlen = cls_num
#
#     val_train_vectors = np.array(val_train_vectors)
#     val_test_vectors = np.array(val_test_vectors)
#     val_train_labels1 = np.array(val_train_labels1)
#     val_test_labels1 = np.array(val_test_labels1)
#     train_labels1 = np.array(train_labels1)
#     test_labels1 = np.array(test_labels1)
#     print(val_train_vectors.shape)
#     print(val_test_vectors.shape)
#     print(val_train_labels1.shape)
#     print(val_test_labels1.shape)
#     print(train_labels1.shape)
#     print(test_labels1.shape)
#
#     train_vectors = val_train_vectors
#     test_vectors = val_test_vectors
#     val_train_labels = val_train_labels1
#     val_test_labels = val_test_labels1
#     train_labels = train_labels1
#     test_labels = test_labels1








#dump word embedding for write results
pickle_file_path = './{}cls_elmo_token38_val.pickle'.format(cls_num)
with open(pickle_file_path, 'wb') as pickle_file:
    pickle.dump([train_vectors, test_vectors], pickle_file)

#dump data for running model
model_data_path = './{}cls_elmo_embeddding.pickle'.format(cls_num)
with open(model_data_path,'wb') as f:
    pickle.dump([train_vectors,test_vectors,val_train_labels,val_test_labels,train_labels,test_labels,maxlen,embedding_dim],f)