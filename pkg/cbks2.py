import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.metrics import auc as calculate_auc
from sklearn.metrics import accuracy_score
from scipy.stats.stats import pearsonr


def r2_score(x,y):
    pcc, _ = pearsonr(x,y)
    return pcc**2


def prc_auc_score(y_true, y_score):
    precision, recall, threshold  = precision_recall_curve(y_true, y_score)
    auc = calculate_auc(recall, precision)
    return auc


class T3Performance(tf.keras.callbacks.Callback):
    def __init__(self, train_data, valid_data, MASK=-1, patience=5, criteria='val_loss', metric='ROC', last_avf=None, verbose=0):
        super(T3Performance, self).__init__()
        sp = ['val_loss', 'val_auc', 'val_acc']
        assert criteria in sp, 'not support %s ! only %s' % (criteria, sp)
        self.x, self.y = train_data
        self.x_val_both, self.y_val_both = valid_data
        self.x_val, self.x_val_dual = self.x_val_both
        self.y_val, self.y_val_dual = self.y_val_both
        self.last_avf = last_avf

        self.history = {'loss': [],
                        'val_loss': [],
                        'auc': [],
                        'val_auc': [],
                        'val_dual_auc': [],
                        'epoch': []}
        self.MASK = MASK
        self.patience = patience
        self.best_weights = None
        self.criteria = criteria
        self.metric = metric
        self.best_epoch = 0
        self.verbose = verbose

    def sigmoid(self, x):
        s = 1/(1+np.exp(-x))
        return s


    def roc_auc(self, y_true, y_pred):
        if self.last_avf is None:
            y_pred_logits = self.sigmoid(y_pred)
        else:
            y_pred_logits = y_pred

        N_classes = y_pred_logits.shape[1]

        aucs = []
        for i in range(N_classes):
            y_pred_one_class = y_pred_logits[:,i]
            y_true_one_class = y_true[:, i]
            mask = ~(y_true_one_class == self.MASK)
            try:
                if self.metric == 'ROC':
                    auc = roc_auc_score(y_true_one_class[mask], y_pred_one_class[mask]) # ROC_AUC
                elif self.metric == 'PRC':
                    auc = prc_auc_score(y_true_one_class[mask], y_pred_one_class[mask]) # PRC_AUC
                elif self.metric == 'ACC':
                    auc = accuracy_score(y_true_one_class[mask], np.round(y_pred_one_class[mask])) # ACC
            except:
                auc = np.nan
            aucs.append(auc)
        return aucs


    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        if self.criteria == 'val_loss':
            self.best = np.Inf
        else:
            self.best = -np.Inf


    def on_epoch_end(self, epoch, logs={}):

        y_pred = self.model.predict(self.x)
        roc_list = self.roc_auc(self.y, y_pred)
        roc_mean = np.nanmean(roc_list)

        y_pred_val = self.model.predict(self.x_val)
        roc_val_list = self.roc_auc(self.y_val, y_pred_val)
        roc_val_mean = np.nanmean(roc_val_list)

        proba_dual = self.model.predict(self.x_val_dual)
        pre_dual = np.round(proba_dual)

        self.history['loss'].append(logs.get('loss'))
        self.history['val_loss'].append(logs.get('val_loss'))
        self.history['auc'].append(roc_mean)
        self.history['val_auc'].append(roc_val_mean)
        self.history['epoch'].append(epoch)

        eph = str(epoch+1).zfill(4)
        self.eph = eph

        if self.criteria == 'val_loss':
            current = logs.get('val_loss')
            if current <= self.best and epoch >= 20:
                self.best = current
                self.wait = 0
                self.best_weights = self.model.get_weights()
                self.best_epoch = epoch
            else:
                self.wait += 1
                if epoch < 20:
                    self.wait = 0
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
                    print('\n storing model weights.')
                    self.model.set_weights(self.best_weights)
        else:
            current = roc_val_mean
            if current >= self.best and epoch >= 20:
                self.best = current
                self.wait = 0
                self.best_weights = self.model.get_weights()
                self.best_epoch = epoch
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
                    print('\n storing model weights.')
                    self.model.set_weights(self.best_weights)


    def on_train_end(self, logs=None):
        self.model.set_weights(self.best_weights)
        if self.stopped_epoch > 0:
            print('\nEpoch %05d: early stopping' % (self.stopped_epoch + 1))


    def evaluate(self, testX, testY):
        y_pred = self.model.predict(testX)
        roc_list = self.roc_auc(testY, y_pred)
        return roc_list
