from keras.models import Sequential, load_model
from keras.callbacks import History, EarlyStopping, Callback
from keras.layers import LSTM
from keras.layers.core import Dense, Activation, Dropout
import numpy as np
import os
import logging

from tsai.all import *

from fastai.callback.all import *

import json

# suppress tensorflow CPU speedup warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logger = logging.getLogger('telemanom')


class Model:
    def __init__(self, config, run_id, channel):
        """
        Loads/trains RNN and predicts future telemetry values for a channel.

        Args:
            config (obj): Config object containing parameters for processing
                and model training
            run_id (str): Datetime referencing set of predictions in use
            channel (obj): Channel class object containing train/test data
                for X,y for a single channel

        Attributes:
            config (obj): see Args
            chan_id (str): channel id
            run_id (str): see Args
            y_hat (arr): predicted channel values
            model (obj): trained RNN model for predicting channel values
        """

        self.config = config
        self.chan_id = channel.id
        self.run_id = run_id
        self.y_hat = np.array([])
        self.model = None
        self.reg = None

        if not self.config.train:
            try:
                self.load()
            except FileNotFoundError:
                path = os.path.join('data', self.config.use_id, 'models',
                                    self.chan_id + '.h5')
                logger.warning('Training new model, couldn\'t find existing '
                               'model at {}'.format(path))
                self.train_new(channel)
                self.save()
        else:
            self.train_new(channel)
            self.save()
            
    def load(self):
        if self.config.arch == 'classic':
            self.load_classic()
        else:
            self.load_tsai()

    def load_classic(self):
        """
        Load model for channel.
        """

        logger.info('Loading pre-trained model')
        self.model = load_model(os.path.join('data', self.config.use_id,
                                             'models', self.chan_id + '.h5'))
                                             
    def load_tsai(self):
        self.reg = load_learner("models/reg_{}_{}.pkl".format(self.chan_id, self.config.arch))
                                             
    def train_new(self, channel):
        if self.config.arch == 'classic':
            self.train_new_classic(channel)
        else:
            self.train_new_tsai(channel)
            
    def tsai_arch_to_model(self, arch):
        if arch == 'mWDNPlus': return mWDNPlus
        elif arch == 'MLP': return MLP
        elif arch == 'gMLP': return gMLP
        elif arch == 'TSTPlus': return TSTPlus
        elif arch == 'TransformerLSTMPlus': return TransformerLSTMPlus
        elif arch == 'LSTMAttentionPlus': return LSTMAttentionPlus
        elif arch == 'GRUAttentionPlus': return GRUAttentionPlus
        elif arch == 'FCNPlus': return FCNPlus
        elif arch == 'ResNetPlus': return ResNetPlus
        elif arch == 'XceptionTimePlus': return XceptionTimePlus
        elif arch == 'InceptionTimeXLPlus': return InceptionTimeXLPlus
        elif arch == 'LSTMPlus': return LSTMPlus
        elif arch == 'LSTM_FCNPlus': return LSTM_FCNPlus
        elif arch == 'GRUPlus': return GRUPlus
        elif arch == 'OmniScaleCNN': return OmniScaleCNN
        elif arch == 'MultiInceptionTimePlus': return MultiInceptionTimePlus
        else: return None
          
    def train_new_tsai(self, channel):
        print("training tsai")
        
        monitor = 'valid_loss' # for everything else
        seed = 42
        lr_max = 3e-4

        cbs=[EarlyStoppingCallback(monitor=monitor, min_delta=self.config.min_delta, patience=self.config.patience),
             SaveModelCallback(monitor=monitor, min_delta=self.config.min_delta),
             CSVLogger(fname=f"{self.config.arch}_{channel.id}_log.csv"),
             ReduceLROnPlateau(monitor=monitor)]
        tfms = [None, TSRegression()]
        batch_tfms = TSStandardize(by_sample=True)

        arch = self.tsai_arch_to_model(self.config.arch)
        arch_config = json.loads(self.config.arch_args)
        
        X_transposed = np.transpose(channel.X_train, (0, 2, 1))
        
        splits = TimeSplitter(valid_size=self.config.validation_split, show_plot=False)(channel.y_train)
        dsets = TSDatasets(X_transposed, channel.y_train, tfms=tfms, splits=splits)
        
        bs = self.config.lstm_batch_size
        dls   = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[bs, bs*2], batch_tfms=batch_tfms)
        
        model = create_model(arch, dls=dls, arch_config=arch_config)
        self.reg = ts_learner(dls, model,  metrics=rmse, path='models')
        
        assert(self.reg != None)
        
        if not hasattr(self.reg.model,'head'):
            self.reg.model.head = nn.Linear(256, 10)
        
        self.reg.fit_one_cycle(self.config.epochs, lr_max, cbs=cbs)
    
    def train_new_tsai_old(self, channel):
        print("training tsai")
        
        monitor = 'valid_loss' # for everything else

        cbs=[EarlyStoppingCallback(monitor=monitor, min_delta=self.config.min_delta, patience=self.config.patience),
             SaveModelCallback(monitor=monitor, min_delta=self.config.min_delta),
             CSVLogger(fname=f"{self.config.arch}_{channel.id}_log.csv"),
             ReduceLROnPlateau(monitor=monitor)]
        tfms = [None, TSRegression()]
        batch_tfms = TSStandardize(by_sample=True)

        arch_config = json.loads(self.config.arch_args)

        #loss_func = nn.MSELoss()
        loss_func = None
        
        seed = 42
        lr_max = 3e-4

        splits = TimeSplitter(valid_size=self.config.validation_split, show_plot=False)(channel.y_train)
        
        
        self.reg = TSForecaster(channel.X_train, 
                          channel.y_train, 
                          path='models', 
                          arch=self.config.arch,
                          arch_config=arch_config,
                          tfms=tfms, 
                          batch_tfms=batch_tfms, 
                          batch_size=self.config.lstm_batch_size,
                          patch_len=self.config.lstm_batch_size,
                          metrics=rmse,
                          loss_func=loss_func,
                          splits=splits,
                          seed = seed,
                          verbose=True)
                          
        assert(self.reg != None)
        
        print(self.reg.model)
        
        self.reg.fit_one_cycle(self.config.epochs, lr_max, cbs=cbs)
            

    def train_new_classic(self, channel):
        """
        Train LSTM model according to specifications in config.yaml.

        Args:
            channel (obj): Channel class object containing train/test data
                for X,y for a single channel
        """

        cbs = [History(), EarlyStopping(monitor='val_loss',
                                        patience=self.config.patience,
                                        min_delta=self.config.min_delta,
                                        verbose=0)]

        self.model = Sequential()

        self.model.add(LSTM(
            self.config.layers[0],
            input_shape=(None, channel.X_train.shape[2]),
            return_sequences=True))
        self.model.add(Dropout(self.config.dropout))

        self.model.add(LSTM(
            self.config.layers[1],
            return_sequences=False))
        self.model.add(Dropout(self.config.dropout))

        self.model.add(Dense(
            self.config.n_predictions))
        self.model.add(Activation('linear'))

        self.model.compile(loss=self.config.loss_metric,
                           optimizer=self.config.optimizer)

        self.model.fit(channel.X_train,
                       channel.y_train,
                       batch_size=self.config.lstm_batch_size,
                       epochs=self.config.epochs,
                       validation_split=self.config.validation_split,
                       callbacks=cbs,
                       verbose=True)

    def save(self):
        if self.config.arch == 'classic':
            self.save_classic()
        else:
            self.save_tsai()
            
    def save_classic(self):
        """
        Save trained model.
        """

        self.model.save(os.path.join('data', self.run_id, 'models',
                                     '{}.h5'.format(self.chan_id)))
                                     
    def save_tsai(self):
        assert(self.reg != None)
        self.reg.export("reg_{}_{}.pkl".format(self.chan_id, self.config.arch))
        
    def aggregate_predictions(self, y_hat_batch, method='first'):
        self.aggregate_predictions_new(y_hat_batch, method)
        
    def aggregate_predictions_new(self, y_hat_batch, method='first'):
        """
        Aggregates predictions for each timestep. When predicting n steps
        ahead where n > 1, will end up with multiple predictions for a
        timestep.
        """
        
        #import pdb; pdb.set_trace()
        
        agg_y_hat_batch = np.array([])
        
        for t in range(len(y_hat_batch)):

            start_idx = t - self.config.n_predictions
            start_idx = start_idx if start_idx >= 0 else 0
            
            # For some reason some models return predictions as array of array - flatten them
            b = [np.array(arr).flatten() for arr in y_hat_batch[start_idx:t+1]]
            y_hat_t = np.flipud(b).diagonal()
            
            if method == 'first':
                agg_y_hat_batch = np.append(agg_y_hat_batch, [y_hat_t[0]])
            elif method == 'mean':
                agg_y_hat_batch = np.append(agg_y_hat_batch, np.mean(y_hat_t))
                            
        agg_y_hat_batch = agg_y_hat_batch.reshape(len(agg_y_hat_batch), 1)
        
        self.y_hat = np.append(self.y_hat, agg_y_hat_batch)
        

    def aggregate_predictions_old(self, y_hat_batch, method='first'):
        """
        Aggregates predictions for each timestep. When predicting n steps
        ahead where n > 1, will end up with multiple predictions for a
        timestep.

        Args:
            y_hat_batch (arr): predictions shape (<batch length>, <n_preds)
            method (string): indicates how to aggregate for a timestep - "first"
                or "mean"
        """

        agg_y_hat_batch = np.array([])

        for t in range(len(y_hat_batch)):

            start_idx = t - self.config.n_predictions
            start_idx = start_idx if start_idx >= 0 else 0

            # predictions pertaining to a specific timestep lie along diagonal
            y_hat_t = np.flipud(y_hat_batch[start_idx:t+1]).diagonal()

            if method == 'first':
                agg_y_hat_batch = np.append(agg_y_hat_batch, [y_hat_t[0]])
            elif method == 'mean':
                agg_y_hat_batch = np.append(agg_y_hat_batch, np.mean(y_hat_t))

        agg_y_hat_batch = agg_y_hat_batch.reshape(len(agg_y_hat_batch), 1)
        self.y_hat = np.append(self.y_hat, agg_y_hat_batch)
        
    def predict(self, X_test_batch):
        if self.config.arch == 'classic':
            return self.predict_classic(X_test_batch)
        else:
            return self.predict_tsai(X_test_batch)
    
    def predict_classic(self, X_test_batch):
        return self.model.predict(X_test_batch)
        
    def predict_tsai(self, X_test_batch):
        X_transposed = np.transpose(X_test_batch, (0, 2, 1))
        raw_preds, target, preds = self.reg.get_X_preds(X_transposed)
        
        return preds

    def batch_predict(self, channel):
        """
        Used trained LSTM model to predict test data arriving in batches.

        Args:
            channel (obj): Channel class object containing train/test data
                for X,y for a single channel

        Returns:
            channel (obj): Channel class object with y_hat values as attribute
        """

        num_batches = int((channel.y_test.shape[0] - self.config.l_s)
                          / self.config.batch_size)
        if num_batches < 0:
            raise ValueError('l_s ({}) too large for stream length {}.'
                             .format(self.config.l_s, channel.y_test.shape[0]))

        sum = 0
        # simulate data arriving in batches, predict each batch
        for i in range(0, num_batches + 1):
            prior_idx = i * self.config.batch_size
            idx = (i + 1) * self.config.batch_size

            if i + 1 == num_batches + 1:
                # remaining values won't necessarily equal batch size
                idx = channel.y_test.shape[0]

            X_test_batch = channel.X_test[prior_idx:idx]
            y_hat_batch = self.predict(X_test_batch)

            sum += len(y_hat_batch)
            self.aggregate_predictions(y_hat_batch)

        self.y_hat = np.reshape(self.y_hat, (self.y_hat.size,))

        channel.y_hat = self.y_hat

        np.save(os.path.join('data', self.run_id, 'y_hat', '{}_{}.npy'
                             .format(self.config.arch, self.chan_id)), self.y_hat)

        return channel
