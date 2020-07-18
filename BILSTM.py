import tensorflow as tf
import tensorflow_hub as hub
import os
import sys
import logging

import numpy as np
import pandas as pd
import ast
import pickle
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import BatchNormalization,Embedding, Bidirectional, LSTM, Input, Masking,Dropout,Flatten,Activation,RepeatVector,Permute,concatenate,Add,Reshape,Dense,Attention,AdditiveAttention,GRU,LayerNormalization
from tensorflow.keras import Model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import Callback
from sklearn.utils import class_weight
from tensorflow.keras import optimizers
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
_EPSILON = 1e-7
hidden_dim = 512
batch_size = 914
nb_epoch = 35
if_bert = False
if_elmo = True
cls_num = 1
if_ensemble = False
if_ldl = True

def create_model(maxlen, embedding_dim):
    sequence = Input(shape=(maxlen, embedding_dim,), dtype='float32')
    # bnl = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones',beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)(sequence)
    embedded = Masking(mask_value=0.0)(sequence)
    # embedded = Attention(use_scale=True,causal=True)([embedded, embedded])
    hidden = Bidirectional(LSTM(hidden_dim // 2, return_sequences=True, recurrent_dropout=0.25))(embedded)
    # hidden = Bidirectional(LSTM(hidden_dim // 2, return_sequences=True, recurrent_dropout=0.25,kernel_regularizer=tf.keras.regularizers.l2(1e-2)))(hidden)
    hidden = Bidirectional(LSTM(hidden_dim // 2, return_sequences=True, recurrent_dropout=0.25))(hidden)
    # hidden=LayerNormalization(
    #     axis=-1, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
    #     gamma_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
    #     beta_constraint=None, gamma_constraint=None, trainable=True, name=None)(hidden)

    Dense1 = Dropout(0.5)(hidden)
    if cls_num == 1:
        output = Dense(cls_num+1, activation='softmax')(Dense1)
    else:
        output = Dense(cls_num, activation='softmax')(Dense1)
    # output=AMSoftmax(3,3,0.35)(hidden)
    model = Model(inputs=sequence, outputs=output)
    model.summary()


    return model

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        #         val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_predict = np.argmax(np.asarray(self.model.predict(self.validation_data[0])), axis=1)
        #         val_targ = self.validation_data[1]
        val_targ = np.argmax(self.validation_data[1], axis=1)
        _val_f1 = f1_score(val_targ, val_predict, average='micro')
        #         _val_recall = recall_score(val_targ, val_predict)
        #         _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        #         self.val_recalls.append(_val_recall)
        #         self.val_precisions.append(_val_precision)
        #         print('— val_f1: %f — val_precision: %f — val_recall %f' %(_val_f1, _val_precision, _val_recall))
        print(' — val_f1:', _val_f1)
        return _val_f1
def micro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='micro')


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    tf.executing_eagerly()

    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))


    #load model data
    logging.info('loading data...')
    if if_elmo:
        model_data_path = './{}cls_elmo_embeddding.pickle'.format(cls_num)

    if if_bert:
        model_data_path = './{}cls_bert_embeddding.pickle'.format(cls_num)
    train_vectors, test_vectors, val_train_labels, val_test_labels, train_labels, test_labels, maxlen, embedding_dim = pickle.load(open(model_data_path, 'rb'))

    print(train_vectors.shape)
    print(test_vectors.shape)
    print(val_train_labels.shape)
    print(val_test_labels.shape)



    #training
    sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-06, decay=0.0)
    rms = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)

    if if_ensemble:
        model1 = KerasClassifier(build_fn=create_model(maxlen, embedding_dim),  verbose=0,epochs=nb_epoch, batch_size=batch_size,validation_data=[test_vectors, val_test_labels],class_weight='auto')
        model2 = KerasClassifier(build_fn=create_model(maxlen, embedding_dim),  verbose=0,epochs=nb_epoch, batch_size=batch_size,validation_data=[test_vectors, val_test_labels],class_weight='auto')
        model3 = KerasClassifier(build_fn=create_model(maxlen, embedding_dim),  verbose=0,epochs=nb_epoch, batch_size=batch_size,validation_data=[test_vectors, val_test_labels],class_weight='auto')
        ensemble_clf = VotingClassifier(estimators=[
            ('model1', model1), ('model2', model2), ('model3', model3)
        ], voting='soft')
        ensemble_clf.fit(train_vectors, val_train_labels)

    else:
        # with tf.device('/cpu:0'):
        #     model = create_model(maxlen, embedding_dim)
        #
        # parallel_model = multi_gpu_model(model, gpus=4)
        # parallel_model.compile(optimizer=rms, loss="mean_squared_error", metrics=['accuracy'])
        # parallel_model.fit(train_vectors, val_train_labels, epochs=nb_epoch, batch_size=batch_size,validation_data=[test_vectors, val_test_labels],class_weight='auto')
        model = create_model(maxlen, embedding_dim)
        model.compile(optimizer=adam, loss="kullback_leibler_divergence", metrics=['accuracy'])
        model.fit(train_vectors, val_train_labels, epochs=nb_epoch,batch_size=batch_size,validation_data=[test_vectors, val_test_labels])

    if if_elmo:
        model.save('{}cls_elmo_mode'.format(cls_num))
        # tf.keras.models.save_model(
        #     model,
        #     '{}cls_elmo_mode'.format(cls_num),
        #     overwrite=True,
        #     include_optimizer=True,
        #     save_format='tf',
        #     signatures=None,
        #     options=None
        # )
        # tf.keras.experimental.export_saved_model(model, '{}cls_elmo_mode'.format(cls_num))
    if if_bert:
        model.save('{}cls_bert_mode'.format(cls_num))


    #evaluate

    print(model.predict(test_vectors))
    if if_ensemble:
        test_pred = ensemble_clf.predict(test_vectors).argmax(-1)
    else:
        test_pred = model.predict(test_vectors).argmax(-1)
    print("test_pred")
    print(test_pred)
    print(test_pred.shape)
    y_true, y_pred = [], []
    for i, labels in enumerate(test_labels):
        for j, label in enumerate(labels):
            y_true.append(label[1])
    for i in range(len(y_true)):
        y_true[i] = int(y_true[i])
    y_pred = []
    for i, labels in enumerate(test_labels):
        for j, label in enumerate(labels):
            if j< len(test_pred[i]):
                y_pred.append(test_pred[i][j])
            else:
                y_pred.append(0)



    print("y_pred")
    print(y_pred)
    print("len(y_pred)")
    print(len(y_pred))
    print("Y-true")
    print(y_true)
    print("len(y_true)")
    print(len(y_true))
    for i in y_true:
        i = int(i)
    logging.info('classes f1_score: ' + str(f1_score(y_true, y_pred, average=None)))
    logging.info('classes precision_score: ' + str(precision_score(y_true, y_pred, average=None)))
    logging.info('classes recall_score: ' + str(recall_score(y_true, y_pred, average=None)))

    logging.info('f1_score: ' + str(f1_score(y_true, y_pred, average='micro')))
    logging.info('precision_score: ' + str(precision_score(y_true, y_pred, average='micro')))
    logging.info('recall_score: ' + str(recall_score(y_true, y_pred, average='micro')))

    logging.info('f1_score: ' + str(f1_score(y_true, y_pred, average='macro')))
    logging.info('precision_score: ' + str(precision_score(y_true, y_pred, average='macro')))
    logging.info('recall_score: ' + str(recall_score(y_true, y_pred, average='macro')))

    logging.info('f1_score: ' + str(f1_score(y_true, y_pred, average='weighted')))
    logging.info('precision_score: ' + str(precision_score(y_true, y_pred, average='weighted')))
    logging.info('recall_score: ' + str(recall_score(y_true, y_pred, average='weighted')))

    logging.info('accuracy_score: ' + str(accuracy_score(y_true, y_pred)))

    logging.info(precision_recall_fscore_support(y_true, y_pred, beta=1, average='micro'))
    logging.info(classification_report(y_true, y_pred))
