import time

import tensorflow as tf

from LatentSpacePhysics.src.nn.stages import *
from LatentSpacePhysics.src.nn.helpers import *
from LatentSpacePhysics.src.nn.losses import *
from LatentSpacePhysics.src.nn.callbacks import LossHistory
from LatentSpacePhysics.src.nn.arch.architecture import Network
from LatentSpacePhysics.src.util import array
from LatentSpacePhysics.src.nn.lstm import error_classification

from ops import *
from math import floor

import keras
from keras.optimizers import Adam
from keras import objectives
from keras.layers import *
from keras.layers.pooling import AveragePooling2D, MaxPooling2D
from keras.models import Model, save_model, load_model
from keras.callbacks import Callback
from keras.regularizers import l1_l2, l2
from keras.utils import multi_gpu_model
import keras.backend as K

from keras_models_general import model_to_json

#=====================================================================================
class Prediction(Network):
    #---------------------------------------------------------------------------------
    def _init_vars(self, config, **kwargs):
        self.input_shape = kwargs.get("input_shape", (2, 32))

        # Network Parameters
        self.stateful = kwargs.get("stateful", False)
        self.in_out_states = kwargs.get("in_out_states", False)

        self.states = kwargs.get("states", [[None, None], [None, None], [None, None]])
        self.return_state = self.in_out_states

        self.use_attention = False
        self.use_bidirectional = False
        self.lstm_activation = "tanh"

        self.use_time_conv_decoder = True
        self.time_conv_decoder_filters = 256
        self.time_conv_decoder_depth = 1

        self.adam_epsilon = None
        self.adam_learning_rate = 0.000126
        self.adam_lr_decay = 0.000334
        self.use_bias = True

        self.kernel_regularizer = None
        self.recurrent_regularizer = None
        self.bias_regularizer = None
        self.activity_regularizer = None

        self.dropout=0.0
        self.recurrent_dropout=0.0

        self.b_num = config.batch_size
        self.z_num = config.z_num
        self.w_num = self.input_shape[0] 
        self.out_w_num = 1

        if hasattr(config, 'encoder_lstm_neurons'):
            self.encoder_lstm_neurons = config.encoder_lstm_neurons
        else:
            self.encoder_lstm_neurons = 512
        if hasattr(config, 'decoder_lstm_neurons'):
            self.decoder_lstm_neurons = config.decoder_lstm_neurons
        else:
            self.decoder_lstm_neurons = 512

        # Loss Setup
        self.set_loss(loss="mse")
        self.l1_reg = kwargs.get("l1_reg", 0.0)
        self.l2_reg = kwargs.get("l2_reg", 0.0)
        self.tensorflow_seed = kwargs.get("tensorflow_seed", 4)
        self.model = None
        tf.set_random_seed(self.tensorflow_seed)
        self.gpus = [ int(gpu.strip()) for gpu in config.gpu_id.split(",")]
        print("Using GPUs: {}".format(self.gpus))
        self.parallel_model = None

        # Trainer Variables
        self.config = config
        self.arch = config.arch
        self.is_3d = config.is_3d

        self.optimizer = config.optimizer
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        self.model_dir = config.model_dir
        self.load_path = config.load_path

        self.dataset = config.dataset

    #---------------------------------------------------------------------------------
    def set_states(self, states):
        self.states = states

    #---------------------------------------------------------------------------------
    def set_loss(self, loss):
        self.loss = loss
        self.metrics = []

    #---------------------------------------------------------------------------------
    def _init_optimizer(self, epochs=1):
        self.optimizer = Adam(lr=self.adam_learning_rate, epsilon=self.adam_epsilon, decay=self.adam_lr_decay)
        return self.optimizer

    #---------------------------------------------------------------------------------
    def _fix_output_dimension(self, x):
        # https://github.com/keras-team/keras/issues/7961
        pre_shape = x.shape.as_list()
        pre_shape[1] = self.out_w_num
        x.set_shape(pre_shape)
        return x

    #---------------------------------------------------------------------------------
    def _build_model(self):
        if self.stateful:
            pred_input = Input(batch_shape=(self.b_num,) + self.input_shape, dtype="float32", name='Temp_Prediction_Input') # (self.time_steps, self.data_dimension)
        else:
            pred_input = Input(shape=self.input_shape, dtype="float32", name='Temp_Prediction_Input0') # (self.time_steps, self.data_dimension)
            if self.in_out_states:
                state_input_0_0 = Input(shape=(self.encoder_lstm_neurons,), dtype="float32", name='State00_Prediction_Input') 
                state_input_0_1 = Input(shape=(self.encoder_lstm_neurons,), dtype="float32", name='State01_Prediction_Input') 
                state_input_1_0 = Input(shape=(self.decoder_lstm_neurons,), dtype="float32", name='State10_Prediction_Input') 
                state_input_1_1 = Input(shape=(self.decoder_lstm_neurons,), dtype="float32", name='State11_Prediction_Input') 

        lstm_layer = []

        x = pred_input
        lstm_temp = LSTM(units=self.encoder_lstm_neurons,
                        dropout=self.dropout,
                        recurrent_dropout=self.recurrent_dropout,
                        return_sequences=False,
                        go_backwards=True,
                        stateful=self.stateful,
                        return_state=self.return_state,
                        name="TempPred_0"
                        )
        lstm_layer.append(lstm_temp)
        if self.in_out_states:
            x, self.states[0][0], self.states[0][1] = lstm_layer[-1](x, initial_state=[state_input_0_0, state_input_0_1])
        else:
            x = lstm_layer[-1](x)

        x = RepeatVector(self.out_w_num)(x)

        lstm_temp = LSTM(units=self.decoder_lstm_neurons,
                        dropout=self.dropout,
                        recurrent_dropout=self.recurrent_dropout,
                        return_sequences=True,
                        go_backwards=False,
                        stateful=self.stateful,
                        return_state=self.return_state,
                        name="TempPred_1"
                        )
        lstm_layer.append(lstm_temp)
        if self.in_out_states:
            x, self.states[1][0], self.states[1][1] = lstm_layer[-1](x, initial_state=[state_input_1_0, state_input_1_1])
        else:
            x = lstm_layer[-1](x)

        x = self._fix_output_dimension(x)
        
        if self.use_time_conv_decoder:
            for i in range(self.time_conv_decoder_depth):
                x = Conv1D(filters=self.time_conv_decoder_filters, kernel_size=1, name="TempPred_{}".format(2+i))(x)
                x = LeakyReLU(0.3)(x)
            x = Conv1D(filters=self.z_num, kernel_size=1, name="TempPred_{}".format(2+self.time_conv_decoder_depth))(x)

        x = self._fix_output_dimension(x)

        outputs = [x]
        if self.in_out_states:
            outputs.append(self.states[0][0])
            outputs.append(self.states[0][1])
            outputs.append(self.states[1][0])
            outputs.append(self.states[1][1])

        inputs = [pred_input]
        if self.in_out_states:
            inputs.append(state_input_0_0)
            inputs.append(state_input_0_1)
            inputs.append(state_input_1_0)
            inputs.append(state_input_1_1)

        if len(self.gpus) > 1:
            with tf.device('/cpu:0'):
                self.model = Model(name="Prediction", inputs=inputs, outputs=outputs)
        else:
            self.model = Model(name="Prediction", inputs=inputs, outputs=outputs)

    #---------------------------------------------------------------------------------
    def _inner_RNN_layer(self, use_gru, output_dim, go_backwards, return_sequences, return_state):
        activation=self.lstm_activation #def: tanh
        recurrent_activation='hard_sigmoid' #def: hard_sigmoid

        kernel_regularizer = l2(l=self.kernel_regularizer) if self.kernel_regularizer is not None else None
        recurrent_regularizer = l2(l=self.recurrent_regularizer) if self.recurrent_regularizer is not None else None
        bias_regularizer = l2(l=self.bias_regularizer) if self.bias_regularizer is not None else None
        activity_regularizer = l2(l=self.activity_regularizer) if self.activity_regularizer is not None else None

        if use_gru:
            return GRU( units=output_dim,
                        stateful=self.stateful,
                        go_backwards=go_backwards,
                        return_sequences=return_sequences,
                        activation=activation, #def: tanh
                        recurrent_activation=recurrent_activation, #def: hard_sigmoid
                        dropout=self.dropout, #def: 0.
                        recurrent_dropout=self.recurrent_dropout,  #def: 0.
                        return_state=return_state
                        )
        else:
            return LSTM(units=output_dim,
                        activation=activation, #def: tanh
                        recurrent_activation=recurrent_activation, #def: hard_sigmoid
                        use_bias=self.use_bias,
                        kernel_initializer='glorot_uniform',
                        recurrent_initializer='orthogonal',
                        bias_initializer='zeros',
                        unit_forget_bias=True,

                        kernel_regularizer=kernel_regularizer,
                        recurrent_regularizer=recurrent_regularizer,
                        bias_regularizer=bias_regularizer,
                        activity_regularizer=activity_regularizer,

                        kernel_constraint=None,
                        recurrent_constraint=None,
                        bias_constraint=None,

                        dropout=self.dropout, #def: 0.
                        recurrent_dropout=self.recurrent_dropout,  #def: 0.

                        return_sequences=return_sequences,
                        go_backwards=go_backwards,
                        stateful=self.stateful,
                        return_state=return_state
                        )

    #---------------------------------------------------------------------------------
    def _add_RNN_layer_func(self, previous_layer, output_dim, go_backwards, return_sequences, return_state, bidirectional=False, use_gru=False):
        def _bidirectional_wrapper(use_bidirectional, inner_layer, merge_mode='concat'):
            if use_bidirectional:
                return Bidirectional(layer=inner_layer, merge_mode=merge_mode)
            else:
                return inner_layer

        x = _bidirectional_wrapper(
                use_bidirectional = bidirectional,
                merge_mode = 'sum',
                inner_layer = self._inner_RNN_layer(
                                    use_gru=use_gru,
                                    output_dim=output_dim,
                                    go_backwards=go_backwards,
                                    return_sequences=return_sequences,
                                    return_state=return_state))(previous_layer)
        return x

    #---------------------------------------------------------------------------------
    def _compile_model(self):
        if len(self.gpus) > 1:
            self.parallel_model = multi_gpu_model(self.model, gpus=self.gpus)
            self.parallel_model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        else:
            self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)


    #--------------------------------------------
    # Helper functions
    #--------------------------------------------
    def __get_in_scene_iteration_count(self, sample_count, batch_size):
        return floor((sample_count + 1 - (self.w_num+self.out_w_num)) / batch_size)
    #--------------------------------------------
    def __get_in_scene_iteration_count_dynamic(self, sample_count, in_ts, out_ts, batch_size):
        return floor((sample_count + 1 - (in_ts+out_ts)) / batch_size)
    #--------------------------------------------
    def __generator_nb_batch_samples(self, enc_scenes, batch_size):
        scene_count = len(enc_scenes) # e.g. 10 scenes
        sample_count = enc_scenes[0].shape[0] # with 250 encoded samples each

        in_ts = self.w_num
        out_ts = self.out_w_num
        in_scene_it = self.__get_in_scene_iteration_count_dynamic(sample_count, in_ts, out_ts, batch_size)

        return scene_count * in_scene_it
    #--------------------------------------------
    def __generator_scene_func(self, enc_scenes, batch_size):
        shuffle = self.stateful is False
        scene_count = len(enc_scenes)
        sample_count = enc_scenes[0].shape[0]

        in_scene_iteration = self.__get_in_scene_iteration_count_dynamic(sample_count, self.w_num, self.out_w_num, batch_size)
        print("Scene Count: {}  Sample Count: {}  In-Scene Iteration: {}".format(scene_count, sample_count, in_scene_iteration))

        while 1:
            for i in range(scene_count):
                scene = enc_scenes[i]
                for j in range(in_scene_iteration):
                    enc_data = scene

                    start = j * batch_size
                    end = sample_count#((j+1) * self.batch_size) # - self.out_time_steps
                    X, Y = error_classification.restructure_encoder_data(
                                        data = enc_data[start : end],
                                        time_steps = self.w_num,
                                        out_time_steps = self.out_w_num,
                                        max_sample_count = batch_size)

                    # convert to (#batch, #ts, element_size)
                    X = X.reshape(*X.shape[0:2], -1)
                    Y = Y.reshape(Y.shape[0], self.out_w_num, -1)

                    if shuffle:
                        array.shuffle_in_unison(X, Y)

                    if self.in_out_states:
                        yield [X, np.zeros((self.b_num, self.encoder_lstm_neurons)), np.zeros((self.b_num, self.encoder_lstm_neurons)), np.zeros((self.b_num, self.decoder_lstm_neurons)), np.zeros((self.b_num, self.decoder_lstm_neurons))], [Y, np.zeros((self.b_num, self.encoder_lstm_neurons)), np.zeros((self.b_num, self.encoder_lstm_neurons)), np.zeros((self.b_num, self.decoder_lstm_neurons)), np.zeros((self.b_num, self.decoder_lstm_neurons))]
                    else:
                        yield [X], [Y]

    #---------------------------------------------------------------------------------
    def _train(self, epochs = 5, **kwargs):
        # Arguments
        X = kwargs.get("X")
        Y = kwargs.get("Y")
        train_scenes = kwargs.get("train_scenes", None)
        validation_split = kwargs.get("validation_split")
        callbacks = kwargs.get("callbacks", [])

        # Train
        model = self.model if self.parallel_model is None else self.parallel_model
        batch_size = kwargs.get("batch_size", 8)
        history = keras.callbacks.History()
        history.on_train_begin()

        # Default values for optional parameters
        if validation_split == None:
            validation_split = 0.1

        # Train
        train_generator = None
        validation_generator = None
        train_gen_nb_samples = 0
        val_gen_nb_samples = 0

        if train_scenes is not None:
            # validation split
            validation_scenes = train_scenes[ floor(len(train_scenes) * (1.0 - validation_split)) : ]
            train_scenes = train_scenes[ : floor(len(train_scenes) * (1.0 - validation_split)) ]

            # use generator
            train_gen_nb_samples = self.__generator_nb_batch_samples(train_scenes, batch_size)
            print ("Number of train batch samples per epoch: {}".format(train_gen_nb_samples))
            assert train_gen_nb_samples > 0, ("Batch size is too large for current scene samples/timestep settings. Training by generator not possible. Please adjust the batch size in the 'settings.json' file.")
            train_generator = self.__generator_scene_func(train_scenes, batch_size)

            # validation samples
            val_gen_nb_samples = self.__generator_nb_batch_samples(validation_scenes, batch_size)
            assert val_gen_nb_samples > 0, ("Batch size is too large for current scene samples/timestep settings. Training by generator not possible. Please adjust the batch size in the 'settings.json' file.")
            print ("Number of validation batch samples per epoch: {}".format(val_gen_nb_samples))
            validation_generator = self.__generator_scene_func(validation_scenes, batch_size)

        try:
            trainingDuration = 0.0
            trainStartTime = time.time()
            if self.stateful:
                reset_callback = StatefulResetCallback(model)
                callbacks.append(reset_callback)
                if (train_scenes is None):
                    assert X is not None and Y is not None, ("X or Y is None!")
                    for i in range(epochs):
                        hist = model.fit(
                            X,
                            Y,
                            epochs=1,
                            batch_size=batch_size,
                            shuffle=False,
                            validation_split=validation_split,
                            callbacks=callbacks)
                        history = merge_histories(history, hist)
                        model.reset_states()
                else:
                    for i in range(epochs):
                        hist = model.fit_generator(
                            generator=train_generator,
                            steps_per_epoch=train_gen_nb_samples, # how many batches to draw per epoch
                            epochs = 1,
                            verbose=1,
                            callbacks=callbacks,
                            validation_data=validation_generator,
                            validation_steps=val_gen_nb_samples,
                            class_weight=None,
                            workers=1)
                        history = merge_histories(history, hist)
                        model.reset_states()
            else:
                if self.return_state:
                    reset_callback = StatefulResetCallback(model)
                    callbacks.append(reset_callback)
                if (train_scenes is None):
                    assert X is not None and Y is not None, ("X or Y is None!")
                    history = model.fit(
                        X,
                        Y,
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_split=validation_split,
                        callbacks=callbacks)
                else:
                    history = model.fit_generator(
                        generator=train_generator,
                        steps_per_epoch=train_gen_nb_samples,
                        epochs = epochs,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=validation_generator,
                        validation_steps=val_gen_nb_samples,
                        class_weight=None,
                        workers=1)
            trainingDuration = time.time() - trainStartTime
        except KeyboardInterrupt:
            print("Training duration (s): {}\nInterrupted by user!".format(trainingDuration))
        print("Training duration (s): {}".format(trainingDuration))
        
        return history

    #---------------------------------------------------------------------------------
    def print_summary(self):
        print("Prediction")
        self.model.summary()

    #---------------------------------------------------------------------------------
    def load_model(self, path):
        print("Loading model from {}".format(path))
        temp_model = load_model(path + "/prediction.h5")
        if self.model is None:
            self._build_model()
        self.model.set_weights(temp_model.get_weights())

    #---------------------------------------------------------------------------------
    def save_model(self, path):
        print("Saving model to {}".format(path))
        # serialize model to JSON
        model_to_json(self.model, path + "/prediction.json")
        save_model(self.model, path + "/prediction.h5")

    #---------------------------------------------------------------------------------
    def predict(self, x, batch_size=32):
        return self.model.predict(x, batch_size=batch_size)

from config import get_config
from utils import prepare_dirs_and_logger
from keras_data import BatchManager
import os
from utils import save_image
from LatentSpacePhysics.src.util.requirements import init_packages
from LatentSpacePhysics.src.nn.lstm.sequence_training_data import *
init_packages()

import git

#---------------------------------------------------------------------------------
if __name__ == "__main__":
    config, unparsed = get_config()
    prepare_dirs_and_logger(config)

    config_d = vars(config) if config else {}
    unparsed_d = vars(unparsed) if unparsed else {}

    with open(config.model_dir + "/input_args.json", 'w') as fp:
        json.dump({**config_d, **unparsed_d}, fp)

    # create GIT file
    repo = git.Repo(search_parent_directories=False)
    open("{}/{}".format(config.model_dir, repo.head.object.hexsha), "w") 

    # Transfer data to local vars
    batch_num = config.batch_size
    validation_split = 0.1
    epochs = config.epochs