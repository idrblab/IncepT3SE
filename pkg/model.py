import os
import tensorflow as tf
import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, ClassifierMixin
from .cbks2 import T3Performance
from .net2 import T3Net
import warnings
warnings.filterwarnings("ignore")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class T3SEClassEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 n_outputs,
                 fmap_shape1,
                 epochs=1000,
                 dense_layers=[128, 64],
                 dense_avf='relu',
                 batch_size=128,
                 lr=1e-4,
                 decay=0.001,
                 loss='focal_loss',
                 monitor='val_loss',
                 metric='ROC',
                 patience=20,
                 verbose=2,
                 random_state=32,
                 name="T3SE Class Estimator",
                 gpuid="0,1,2,3,4,5,6,7",
                 ):

        self.n_outputs = n_outputs
        self.fmap_shape1 = fmap_shape1
        self.epochs = epochs
        self.dense_layers = dense_layers
        self.dense_avf = dense_avf
        self.batch_size = batch_size
        self.lr = lr
        self.decay = decay
        self.loss = loss
        self.monitor = monitor
        self.metric = metric
        self.patience = patience
        self.verbose = verbose
        self.random_state = random_state
        self.name = name
        self.gpuid = str(gpuid)
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpuid

        np.random.seed(self.random_state)
        tf.compat.v1.set_random_seed(self.random_state)

        model = T3Net(self.fmap_shape1,
                            n_outputs=self.n_outputs,
                            dense_layers=self.dense_layers,
                            dense_avf=self.dense_avf,
                            last_avf='softmax')

        opt = tf.keras.optimizers.Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=self.decay)
        model.compile(optimizer=opt, loss=self.loss, metrics=['accuracy'])
        self._model = model
        print(self)


    def fit(self, X, y, X_valid=None, y_valid=None):
        # Check shape
        if X.ndim != 4:
            raise ValueError("expected dim == 4.")
        w, h, c = X.shape[1:]
        w_, h_, c_ = self.fmap_shape1
        assert (w == w_) & (h == h_) & (c == c_), "Input shape not matched"

        self.X_ = X
        self.y_ = y
        if (X_valid is None) | (y_valid is None):
            X_valid = X
            y_valid = y

        performance = T3Performance((X, y), (X_valid, y_valid),
                                                      patience=self.patience,
                                                      criteria=self.monitor,
                                                      metric=self.metric,
                                                      last_avf="softmax",
                                                      verbose=0, )

        history = self._model.fit(X, y,
                                  batch_size=self.batch_size,
                                  epochs=self.epochs, verbose=self.verbose, shuffle=True,
                                  validation_data=(X_valid[0], y_valid[0]),
                                  callbacks=[performance])

        self._performance = performance
        self.history = history
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        y_prob = self._model.predict(X)
        return y_prob

    def predict(self, X):
        check_is_fitted(self)
        y_pred = np.round(self.predict_proba(X))
        return y_pred

    def score(self, X, y):
        metrics = self._performance.evaluate(X, y)
        return np.nanmean(metrics)
