import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow_hub as hub
import numpy as np
import pickle
import os
cls_num = 1
if_ensemble = False
if_test = False



def write_results(word_id_lsts, words_lsts, e_freq_lsts, write_to, maxlen):
    """
    This function writes results in the format.
    :param word_id_lsts: list of word_ids
    :param words_lsts: list of words
    :param e_freq_lsts: lists of emphasis probabilities
    :param write_to: writing directory
    :return:
    """
    with open(write_to, 'w') as out:
        sentence_id=""
        # a loop on sentences:
        for i in range(len(words_lsts)):
            # a loop on words in a sentence:
            for j in range(len(words_lsts[i])):
                    # writing:
                    if sentence_id == i:
                        to_write = "{}\t{}\t{}\t".format(word_id_lsts[i][j], words_lsts[i][j], e_freq_lsts[i][j])
                        out.write(to_write + "\n")
                    else:
                        out.write("\n")
                        to_write = "{}\t{}\t{}\t".format(word_id_lsts[i][j], words_lsts[i][j], e_freq_lsts[i][j])
                        out.write(to_write + "\n")
                        sentence_id = i

        out.write("\n")
        out.close()


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

def read_data1(filename):
    """
    This function reads the data from .txt file.
    :param filename: reading directory
    :return: lists of word_ids, words, emphasis probabilities, POS tags
    """
    lines = read_lines(filename) + ['']
    word_id_lst, word_id_lsts = [], []
    post_lst, post_lsts = [], []
    sentences = []
    length = []
    for line in lines:
        if line:
            splitted = line.split("\t")
            word_id = splitted[0]
            words = splitted[1]

            word_id_lst.append(word_id)
            post_lst.append(words)
            length.append(len(words))


        elif post_lst:
            word_id_lsts.append(word_id_lst)
            post_lsts.append(post_lst)

            word_id_lst = []
            post_lst = []


    for i in post_lsts:
        sentences.append(" ".join(word for word in i))
    maxlen = max(length)
    return word_id_lsts, post_lsts, sentences,maxlen
def read_lines(filename):
    with open(filename, 'r', encoding='utf-8', errors='ignored') as fp:
        lines = [line.strip() for line in fp]
    return lines





if_bert = True
if_elmo = False
word_id_lsts, post_lsts, bio_lsts, freq_lsts, e_freq_lsts, pos_lsts, sentences = read_data("./train_dev_data/train.txt")

if if_test:
    word_id_lsts1, post_lsts1, sentences1, maxlen= read_data1("./train_dev_data/test_data.txt")
else:
    word_id_lsts1, post_lsts1, bio_lsts1, freq_lsts1, e_freq_lsts1, pos_lsts1, sentences1 = read_data("./train_dev_data/dev.txt")
cls1 = 0
cls2 = 0
cls3 = 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
if if_ensemble:
    cls1 = 1
    cls2 = 5
    cls3 = 4
    cls4 = 3
    cls5 = 2
    model = load_model('./{}cls_elmo_mode.h5'.format(cls1))
    path = './{}cls_elmo_token38_val.pickle'.format(cls1)

else:
    cls1 = cls_num
    if if_elmo:
        model = load_model('./{}cls_elmo_mode.h5'.format(cls_num))
        path = './{}cls_elmo_token38_val.pickle'.format(cls_num)
    if if_bert:
        model = load_model('./{}cls_bert_mode.h5'.format(cls_num))
        path = './{}cls_bert_token38_val.pickle'.format(cls_num)
if if_test:
    test_vectors = pickle.load(open('./test_data_vectors.pickle','rb'))
else:
    train_vectors, test_vectors = pickle.load(open(path,'rb'))
print(test_vectors[1])
print(test_vectors.shape)
maxlen = test_vectors.shape[1]
test_sig = model.predict(test_vectors)
print(test_sig.shape)
print(test_sig[0])
e_freg_lst = []
for i in range(len(word_id_lsts1)):
    e_freg_lst.append([])


print(test_sig[0][0])
for i in range(len(test_sig)):
    for j in range(len(word_id_lsts1[i])):
        if cls1 == 1:
            e_freg_lst[i].append(test_sig[i][j][1]*0.8 + test_sig[i][j][0]*0.2)
        if cls1 == 2:
            e_freg_lst[i].append(test_sig[i][j][0]*0.8 + test_sig[i][j][1]*0.2)
        if cls1 == 5:
            e_freg_lst[i].append(test_sig[i][j][0]*0.4 + test_sig[i][j][1]*0.3 + test_sig[i][j][2]*0.2 + test_sig[i][j][3]*0.1 + test_sig[i][j][4]*0.0)
        if cls1 == 4:
            e_freg_lst[i].append(test_sig[i][j][0] * 0.4 + test_sig[i][j][1] * 0.3 + test_sig[i][j][2] * 0.2 + test_sig[i][j][3] * 0.1)
        if cls1 == 3:
            e_freg_lst[i].append(test_sig[i][j][0] * 0.44 + test_sig[i][j][1] * 0.33 + test_sig[i][j][2] * 0.23)
print(e_freg_lst[1])
if if_ensemble:
    model2 = load_model('./{}cls_elmo_mode.h5'.format(cls2))
    maxlen = test_vectors.shape[1]
    test_sig = model2.predict(test_vectors)
    print(test_sig.shape)
    print(test_sig[0])
    e_freg_lst2 = []
    for i in range(len(word_id_lsts1)):
        e_freg_lst2.append([])

    print(test_sig[0][0])
    for i in range(len(test_sig)):
        for j in range(len(word_id_lsts1[i])):
            if cls2 == 1:
                e_freg_lst2[i].append(test_sig[i][j][1] * 0.8 + test_sig[i][j][0] * 0.2)
            if cls2 == 2:
                e_freg_lst2[i].append(test_sig[i][j][0] * 0.8 + test_sig[i][j][1] * 0.2)
            if cls2 == 5:
                e_freg_lst2[i].append(test_sig[i][j][0] * 0.4 + test_sig[i][j][1] * 0.3 + test_sig[i][j][2] * 0.2 + test_sig[i][j][3] * 0.1 + test_sig[i][j][4] * 0.0)
            if cls2 == 4:
                e_freg_lst2[i].append(test_sig[i][j][0] * 0.4 + test_sig[i][j][1] * 0.3 + test_sig[i][j][2] * 0.2 + test_sig[i][j][3] * 0.1)
            if cls2 == 3:
                e_freg_lst2[i].append(test_sig[i][j][0] * 0.44 + test_sig[i][j][1] * 0.33 + test_sig[i][j][2] * 0.23)

    model3 = load_model('./{}cls_elmo_mode.h5'.format(cls3))

    test_sig = model3.predict(test_vectors)
    print(test_sig.shape)
    print(test_sig[0])
    e_freg_lst3 = []
    for i in range(len(word_id_lsts1)):
        e_freg_lst3.append([])
        for j in range(len(word_id_lsts1[i])):
            e_freg_lst3[i].append(0)

    print(test_sig[0][0])
    for i in range(len(test_sig)):
        for j in range(len(word_id_lsts1[i])):
            if cls3 == 2:
                e_freg_lst3[i][j] = test_sig[i][j][1]
            if cls3 == 5:
                e_freg_lst3[i][j] = test_sig[i][j][0] * 0.4 + test_sig[i][j][1] * 0.3 + test_sig[i][j][2] * 0.2 + test_sig[i][j][3] * 0.1 + test_sig[i][j][4] * 0.0
            if cls3 == 4:
                e_freg_lst3[i][j] = test_sig[i][j][0] * 0.4 + test_sig[i][j][1] * 0.3 + test_sig[i][j][2] * 0.2 + test_sig[i][j][3] * 0.1
            if cls3 == 3:
                e_freg_lst3[i][j] = test_sig[i][j][0] * 0.44 + test_sig[i][j][1] * 0.33 + test_sig[i][j][2] * 0.23


    model4 = load_model('./{}cls_elmo_mode.h5'.format(cls4))
    test_sig = model4.predict(test_vectors)
    print(test_sig.shape)
    print(test_sig[0])
    e_freg_lst4 = []

    print(test_sig[0][0])
    for i in range(len(test_sig)):
        e_freg_lst4.append([])
        for j in range(len(word_id_lsts1[i])):
            if cls4 == 2:
                e_freg_lst4[i].append(test_sig[i][j][1])
            if cls4 == 5:
                e_freg_lst4[i].append(test_sig[i][j][0] * 0.4 + test_sig[i][j][1] * 0.3 + test_sig[i][j][2] * 0.2 + test_sig[i][j][3] * 0.1 + test_sig[i][j][4] * 0.0)
            if cls4 == 4:
                e_freg_lst4[i].append(test_sig[i][j][0] * 0.4 + test_sig[i][j][1] * 0.3 + test_sig[i][j][2] * 0.2 + test_sig[i][j][3] * 0.1)
            if cls4 == 3:
                e_freg_lst4[i].append(test_sig[i][j][0] * 0.44 + test_sig[i][j][1] * 0.33 + test_sig[i][j][2] * 0.23)

    model5 = load_model('./{}cls_elmo_mode.h5'.format(cls5))

    maxlen = test_vectors.shape[1]
    test_sig = model5.predict(test_vectors)
    print(test_sig.shape)
    print(test_sig[0])
    e_freg_lst5 = []
    for i in range(len(word_id_lsts1)):
        e_freg_lst5.append([])

    print(test_sig[0][0])
    for i in range(len(test_sig)):
        for j in range(len(word_id_lsts1[i])):
            if cls5 == 1:
                e_freg_lst5[i].append(test_sig[i][j][1] * 0.8 + test_sig[i][j][0] * 0.2)
            if cls5 == 2:
                e_freg_lst5[i].append(test_sig[i][j][0] * 0.8 + test_sig[i][j][1] * 0.2)
            if cls5 == 5:
                e_freg_lst5[i].append(test_sig[i][j][0] * 0.4 + test_sig[i][j][1] * 0.3 + test_sig[i][j][2] * 0.2 + test_sig[i][j][3] * 0.1 + test_sig[i][j][4] * 0.0)
            if cls5 == 4:
                e_freg_lst5[i].append(test_sig[i][j][0] * 0.4 + test_sig[i][j][1] * 0.3 + test_sig[i][j][2] * 0.2 + test_sig[i][j][3] * 0.1)
            if cls5 == 3:
                e_freg_lst5[i].append(test_sig[i][j][0] * 0.44 + test_sig[i][j][1] * 0.33 + test_sig[i][j][2] * 0.23)


    print(e_freg_lst[1])
    print(e_freg_lst2[1])
    print(e_freg_lst3[1])
    print(e_freg_lst4[1])
    print(e_freg_lst5[1])
    e_e_freg_lst = []
    for i in range(len(word_id_lsts1)):
        e_e_freg_lst.append([])
        for j in range(len(word_id_lsts1[i])):
            e_e_freg_lst[i].append(0.25*e_freg_lst5[i][j] + 0.25*e_freg_lst2[i][j]+0.25*e_freg_lst3[i][j]+0.25*e_freg_lst4[i][j])
            # e_e_freg_lst[i].append(0.33*e_freg_lst2[i][j] + 0.34* e_freg_lst3[i][j] + 0.33 * e_freg_lst4[i][j])

if if_ensemble:
    write_results(word_id_lsts1, post_lsts1, e_e_freg_lst, "./res/submission.txt", maxlen)
else:
    write_results(word_id_lsts1, post_lsts1, e_freg_lst, "./res/submission.txt", maxlen)
