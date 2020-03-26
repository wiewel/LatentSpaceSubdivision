import sys,inspect,os
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import time

import tensorflow as tf

from keras_models_spatial_multitile import *
from keras_models_temporal import *
from keras_models_general import *

from LatentSpacePhysics.src.nn.stages import *
from LatentSpacePhysics.src.nn.helpers import *
from LatentSpacePhysics.src.nn.losses import *
from LatentSpacePhysics.src.nn.callbacks import LossHistory
from LatentSpacePhysics.src.nn.arch.architecture import Network
from LatentSpacePhysics.src.util.filesystem import make_dir

from ops import *
from os import path

import keras
from keras.optimizers import Adam
from keras import objectives
from keras.layers import *
from keras.layers.pooling import AveragePooling2D, MaxPooling2D
from keras.models import Model, save_model, load_model
from keras.backend import int_shape
from keras.callbacks import Callback
from keras.utils import multi_gpu_model
import keras.backend as K



import json

# Callbacks --------------------------------------------------------------------------------------------------------------------------------------------------
#=====================================================================================
class PlotPredFields(Callback):
    def __init__(self, pred_func, x, func, path, batch_manager, name="Pred"):
        print("PlotPredFields")
        self._pred_func = pred_func
        self._x = x
        self._func = func
        self._counter = 0
        self._path = path
        self.name = name
        self._batch_manager = batch_manager

    def on_epoch_end(self, acc, loss):
        self._y = self._pred_func(self._x)
        self._func(self._y[0], self._counter, self._path, self._batch_manager, self.name+"_vdt1")
        self._func(self._y[1], self._counter, self._path, self._batch_manager, self.name+"_dt2")
        self._counter += 1

#=====================================================================================
class RecursivePrediction_Multitile(Network):
    #---------------------------------------------------------------------------------
    def _init_vars(self, config, settings=None, **kwargs):
        self.input_shape = kwargs.get("input_shape", (4, 64, 64, 64, 3) if config.is_3d else (4, 128, 96, 2))
        
        self.tf_run_options = None

        # Submodel Vars
        self.ae = None
        self.pred = None
        self.latent_compression = None

        self.adam_epsilon = None
        try:
            self.adam_learning_rate = config.lr
        except AttributeError:
            self.adam_learning_rate = 0.001
        try:
            self.adam_lr_decay = config.lr_decay
        except AttributeError:
            self.adam_lr_decay = 0.0005

        self.l1_reg = kwargs.get("l1_reg", 0.0)
        self.l2_reg = kwargs.get("l2_reg", 0.0)
        self.tensorflow_seed = kwargs.get("tensorflow_seed", 4)
        self.model = None
        tf.set_random_seed(self.tensorflow_seed)
        self.gpus = [ int(gpu.strip()) for gpu in config.gpu_id.split(",")]
        print("Using GPUs: {}".format(self.gpus))
        self.parallel_model = None

        self.stateful = kwargs.get("stateful", False)

        self.decode_predictions = kwargs.get("decode_predictions", False)
        self.skip_pred_steps = kwargs.get("skip_pred_steps", False)
        self.init_state_network = kwargs.get("init_state_network", False)
        self.in_out_states = kwargs.get("in_out_states", False)
        self.pred_gradient_loss = kwargs.get("pred_gradient_loss", False)
        self.ls_prediction_loss = kwargs.get("ls_prediction_loss", False)
        self.ls_supervision = kwargs.get("ls_supervision", False)
        self.sqrd_diff_loss = kwargs.get("sqrd_diff_loss", False)
        self.ls_split = kwargs.get("ls_split", 0.0)
        self.sup_param_count = kwargs.get("supervised_parameters", 1)
        self.train_prediction_only = kwargs.get("train_prediction_only", False)

        # TRAINER VARS
        self.config = config
        self.arch = config.arch
        self.is_3d = config.is_3d

        self.optimizer = config.optimizer
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        self.model_dir = config.model_dir
        self.load_path = config.load_path
        self.nn_path = config.nn_path
        self.only_last_prediction = config.only_last_prediction
        if hasattr(config, 'advection_loss'):
            self.advection_loss = config.advection_loss
        else:
            self.advection_loss = 0.0
        if hasattr(config, 'advection_loss_passive_GT'):
            self.advection_loss_passive_GT = config.advection_loss_passive_GT
        else:
            self.advection_loss_passive_GT = False

        self.passive_data_type = "density" if "density" in self.config.data_type else None
        self.passive_data_type = "levelset" if "levelset" in self.config.data_type else self.passive_data_type
        assert len(config.data_type) > 1 and self.passive_data_type == "density" or "levelset", ("No passive data_type found!")

        self.is_train = config.is_train

        self.dataset = config.dataset

        self.b_num = config.batch_size
        self.z_num = config.z_num
        self.w_num = config.w_num

        self.recursive_prediction = self.input_shape[0] - self.w_num
        print("self.recursive_prediction {}".format(self.recursive_prediction))

        self.loss_weights = []

        self.use_inflow = "inflow" in self.config.data_type
        self.input_inflow_shape = list(self.input_shape)
        self.input_inflow_shape[-1] = 1
        self.input_inflow_shape = tuple(self.input_inflow_shape)

        use_density = "density" in self.config.data_type or "levelset" in self.config.data_type
     
        loss_list = [ AE_Loss(sqrd_diff_loss=self.sqrd_diff_loss, density=use_density, is_3d=self.is_3d) ]
        self.loss_weights.append(1.0)
        if self.advection_loss > 0.0:
            loss_list.append("mse")
            self.loss_weights.append(self.advection_loss)

        assert len(loss_list) == len(self.loss_weights)

        self.set_loss(loss=loss_list)

    #---------------------------------------------------------------------------------
    def set_loss(self, loss):
        self.loss = loss
        self.metrics = None

    #---------------------------------------------------------------------------------
    def _init_optimizer(self, epochs=1):
        self.optimizer = Adam(  lr=self.adam_learning_rate,
                                beta_1=0.9,
                                beta_2=0.999,
                                epsilon=self.adam_epsilon, 
                                decay=self.adam_lr_decay,
                                amsgrad=False)
        return self.optimizer

    #---------------------------------------------------------------------------------
    def _create_submodels(self):
        if not self.ae:
            ae_input_shape = list(self.input_shape[-4:] if self.is_3d else self.input_shape[-3:])
            ae_input_shape[-1] = min(4 if self.is_3d else 3, ae_input_shape[-1])
            ae_input_shape = tuple(ae_input_shape)
            self.ae = Autoencoder(config=self.config, input_shape=ae_input_shape, stateful=self.stateful, supervised_parameters=self.sup_param_count) # (b, z, y, x, c) or (b, y, x, c)
            self.ae._build_model()
        if not self.pred:
            self.pred = Prediction(config=self.config, input_shape=(9, self.z_num), stateful=self.stateful, in_out_states=self.in_out_states) # (b, 2, 32)
            self.pred._build_model()
        if not self.latent_compression:
            lc_in = Input(shape=(9, self.z_num), dtype="float32", name='LatentCompressionInput')
            z_tconv_t0 = Conv1D(filters=self.z_num, kernel_size=9, padding="valid")(lc_in)
            z_tconv_t0 = LeakyReLU(0.3)(z_tconv_t0)
            if len(self.gpus) > 1:
                with tf.device('/cpu:0'):
                    self.latent_compression = Model(name="LatentCompression", inputs=[lc_in], outputs=[z_tconv_t0])
            else:
                self.latent_compression = Model(name="LatentCompression", inputs=[lc_in], outputs=[z_tconv_t0])

    #---------------------------------------------------------------------------------
    import copy

    def _build_model(self, **kwargs):
        print("Building Model")
        is_3d = self.is_3d
        velo_dim = 3 if is_3d else 2

        batch_manager = kwargs.get("batch_manager", None)
        if batch_manager is None and self.advection_loss > 0.0:
            print("WARNING: no batch manager found... creating dummy")
            batch_manager = BatchManager(self.config, self.config.input_frame_count, self.config.w_num, data_args_path=kwargs.get("data_args_path", None))

        self._create_submodels()

        enc = self.ae._encoder
        dec = self.ae._decoder

        pred = self.pred.model
        lc = self.latent_compression

        inputs = Input(shape=self.input_shape, dtype="float32", name="Combined_AE_Input_Fields") # (b, input_depth, tiles, y, x, c)

        print("Input shape: {}".format(inputs))

        def global_concat(x, t):
            # first concat x axis for individual rows
            c0 = K.concatenate([x[:, t, 0], x[:, t, 1], x[:, t, 2]], axis=2)
            c1 = K.concatenate([x[:, t, 3], x[:, t, 4], x[:, t, 5]], axis=2)
            c2 = K.concatenate([x[:, t, 6], x[:, t, 7], x[:, t, 8]], axis=2)
            # then concat y axis
            ct = K.concatenate([c0,c1,c2], axis=1)
            return ct

        # global concat
        global_t1 = Lambda(global_concat, arguments={'t': 1})(inputs)

        # loop over tiles -> generate z
        z_time = []
        for t in range(self.config.w_num):
            z_tiles = []
            #   0   1   2
            #   3   4   5
            #   6   7   8
            for i in range(9):
                # individual tile compression
                def slice_0(x, t, i):
                    return x[:, t, i]
                cur_input = Lambda(slice_0, name="Slice_ae_input_{}".format(i), arguments={"t": t, "i": i})(inputs)
                z_tiles.append(enc(cur_input))
            z_time.append(z_tiles)

        z_tconv_t0 = Lambda(lambda x: K.expand_dims(x, axis=1))(z_time[0][0])
        def concat_0(x):
            return K.concatenate([x[0], K.expand_dims(x[1], axis=1)], axis=1)
        for i in range(1,9):
            z_tconv_t0 = Lambda(concat_0, name="Concate_z_enc_{}".format(i))([z_tconv_t0, z_time[0][i]])

        # Use 1D Conv
        z_tconv_t0 = lc([z_tconv_t0])

        # predict t0 -> t1
        z_pred_t1 = pred([z_tconv_t0])
        z_pred_t1 = self.pred._fix_output_dimension(z_pred_t1)

        x_pred_t1 = dec(Reshape((self.z_num,), name="Reshape_xDecPred_{}".format(0))(z_pred_t1))
        # store prediction of t1 in var
        pred_output = x_pred_t1

        # Pad decoded fields to match total field; (b, 3*y, 3*x, c) -> needed for advection
        x_pred_t1 = Lambda(lambda x: tf.pad(x, [[0,0], [self.ae.input_shape[0], self.ae.input_shape[0]], [self.ae.input_shape[1], self.ae.input_shape[1]], [0,0]], "CONSTANT", constant_values=0))(x_pred_t1)

        # apply advection on t0 density with predicted velocity
        if self.advection_loss > 0.0:
            cur_decoded_pred = x_pred_t1

            # 0) get first GT density field that is to be advected (0,1) -> 2 [take 1]
            global_t1_den = Lambda(lambda x: x[..., velo_dim:velo_dim+1], name="gt_passive_{}".format(0))(global_t1)
            global_t1_den = batch_manager.denorm(global_t1_den, self.passive_data_type, as_layer=True)

            # 1) extract velocity array (z,y,x,3) [or (...,2)]
            pred_vel = Lambda(lambda x: x[...,0:velo_dim], name="vel_extract_{}".format(i))(cur_decoded_pred)
    
            # 2) denormalize velocity -> v = keras_data.denorm_vel(v)
            denorm_pred_vel = batch_manager.denorm(pred_vel, "velocity", as_layer=True)

            # 4) use current passive field (z,y,x,1) as advection src
            # 5) call advect(src, v, dt=keras_data.time_step, mac_adv=False, name="density")
            # 6) store as d+1 for usage in next frame -> rec_den
            global_t2_pred_den = Lambda(advect, arguments={'dt': batch_manager.time_step, 'mac_adv': False, 'name': self.passive_data_type}, name="Advect_{}".format(0))( [global_t1_den, denorm_pred_vel] )

            # 6.1) cutoff padding, that was added earlier
            global_t2_pred_den = Lambda(lambda x: x[:,self.ae.input_shape[0]:-self.ae.input_shape[0], self.ae.input_shape[1]:-self.ae.input_shape[1]])(global_t2_pred_den)

            # 7) normalize returned advected passive quantity
            rec_den_norm = batch_manager.norm(global_t2_pred_den, self.passive_data_type, as_layer=True)

            # 8) hand over to loss -> (advect(d^t,v^t), d^t+1)
            adv_output = rec_den_norm

        # decoder loss 
        output_list = [pred_output]
        
        if self.advection_loss > 0.0:
            # adv_output represents density of t2 produced by d2_adv = (v1_pred, d1_gt)
            output_list.append(adv_output)

        # inputs
        input_list = [inputs]

        print("Setup Model")
        if len(self.gpus) > 1:
            with tf.device('/cpu:0'):
                self.model = Model(name="Combined_AE_LSTM", inputs=input_list, outputs=output_list)
        else:
            self.model = Model(name="Combined_AE_LSTM", inputs=input_list, outputs=output_list)

        #self.model.summary()

    #---------------------------------------------------------------------------------
    def _compile_model(self):
        if len(self.gpus) > 1:
            self.parallel_model = multi_gpu_model(self.model, gpus=self.gpus)
            self.parallel_model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics, loss_weights=self.loss_weights, options=self.tf_run_options)
        else:
            self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics, loss_weights=self.loss_weights, options=self.tf_run_options)

    #---------------------------------------------------------------------------------
    def train(self, epochs, **kwargs):
        print("Overwrite of train")
        # Reset all random number generators to given seeds
        np.random.seed(4213)
        tf.set_random_seed(3742)

        # Destroys the current TF graph and creates a new one.
        # Useful to avoid clutter from old models / layers.
        # if self.model is not None:
        #     del self.model
        #     self.model = None
        # K.clear_session()

        # Recompile (in case of updated hyper parameters)
        self._init_optimizer(epochs)
        if not self.model:
            self._build_model(**kwargs)
            self._compile_model()
        # Model Summary
        #self.model.summary()
        self.print_summary()
        self.print_attributes()

        # Train and return History
        history = self._train(epochs, **kwargs)
        return history

    #---------------------------------------------------------------------------------
    def _train(self, epochs = 5, **kwargs):
        # Arguments
        X = kwargs.get("X")
        Y = kwargs.get("Y")
        batch_manager = kwargs.get("batch_manager")
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

        if batch_manager:
            # use generator
            train_gen_nb_samples = batch_manager.steps_per_epoch(batch_size, validation_split, validation=False)
            print ("Number of train batch samples per epoch: {}".format(train_gen_nb_samples))
            assert train_gen_nb_samples > 0, ("Batch size is too large for current scene samples/timestep settings. Training by generator not possible. Please adjust the batch size in the 'settings.json' file.")
            train_generator = batch_manager.generator_ae_tile_sequence(batch_size, validation_split, validation=False, ls_split_loss=self.ls_split > 0.0, advection_loss=self.advection_loss > 0.0)

            # validation samples
            val_gen_nb_samples = batch_manager.steps_per_epoch(batch_size, validation_split, validation=True)
            assert val_gen_nb_samples > 0, ("Batch size is too large for current scene samples/timestep settings. Training by generator not possible. Please adjust the batch size in the 'settings.json' file.")
            print ("Number of validation batch samples per epoch: {}".format(val_gen_nb_samples))
            validation_generator = batch_manager.generator_ae_tile_sequence(batch_size, validation_split, validation=True, ls_split_loss=self.ls_split > 0.0, advection_loss=self.advection_loss > 0.0)

        try:
            trainingDuration = 0.0
            trainStartTime = time.time()
            if self.stateful:
                callbacks.append(StatefulResetCallback(model))
                if (batch_manager is None):
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
                filepath=self.model_dir + "/checkpoint/"
                checkpoint = SaveCheckpoint(filepath, self, monitor="val_loss", verbose=1, save_best_only=True, mode='auto')
                callbacks.append(checkpoint)
                if (batch_manager is None):
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
                        max_queue_size=10)
            trainingDuration = time.time() - trainStartTime
        except KeyboardInterrupt:
            print("Training duration (s): {}\nInterrupted by user!".format(trainingDuration))
        print("Training duration (s): {}".format(trainingDuration))
        
        return history

    #---------------------------------------------------------------------------------
    def print_summary(self):
        self.model.summary()
        with open(path.join(self.config.model_dir, "model_summary.txt"),'w') as msf:
            self.model.summary(print_fn=lambda x: msf.write(x + "\n"))

        from keras.utils.vis_utils import plot_model
        plot_model(self.model, to_file=self.model_dir+"/model_plot.png", show_shapes=True, show_layer_names=True)
        plot_model(self.ae._encoder, to_file=self.model_dir+"/enc_plot.png", show_shapes=True, show_layer_names=True)
        plot_model(self.ae._decoder, to_file=self.model_dir+"/dec_plot.png", show_shapes=True, show_layer_names=True)
        plot_model(self.ae._p_pred, to_file=self.model_dir+"/p_pred_plot.png", show_shapes=True, show_layer_names=True)
        plot_model(self.pred.model, to_file=self.model_dir+"/pred_plot.png", show_shapes=True, show_layer_names=True)
        plot_model(self.latent_compression, to_file=self.model_dir+"/lc_plot.png", show_shapes=True, show_layer_names=True)
        if self.in_out_states:
            plot_model(self.state_init_model, to_file=self.model_dir+"/state_init_plot.png", show_shapes=True, show_layer_names=True)

    #---------------------------------------------------------------------------------
    def load_model(self, path, load_ae=True, load_pred=True, data_args_path=None):
        print("Loading model from {}".format(path))

        self._create_submodels()
        if load_ae:
            self.ae.load_model(path)
        if load_pred:
            if os.path.isfile(path + "/prediction.h5"):
                self.pred.load_model(path)
            else:
                print("WARNING: prediction model could not be loaded, since it was not found at '{}'!".format(path + "/prediction.h5"))

        temp_model = load_model(path + "/lc.h5")
        self.latent_compression.set_weights(temp_model.get_weights())

        if self.model is None:
            self._build_model(data_args_path=data_args_path)

        if os.path.isfile(path + "/state_init.h5"):
            self.state_init_model = load_model(path + "/state_init.h5")

    #---------------------------------------------------------------------------------
    def save_model(self, path):
        print("Saving model to {}".format(path))
        # store weights
        self.model.save_weights(path + "/combined_ae_lstm.h5")
        self.ae.save_model(path)
        self.pred.save_model(path)
        model_to_json(self.latent_compression, path + "/lc.json")
        save_model(self.latent_compression, path + "/lc.h5")
        if self.in_out_states:
            model_to_json(self.state_init_model, path + "/state_init.json")
            save_model(self.state_init_model, path + "/state_init.h5")

    #---------------------------------------------------------------------------------
    def predict(self, x, batch_size=32):
        return self.model.predict(x, batch_size=batch_size)

    #---------------------------------------------------------------------------------
    def ae_predict(self, x, batch_size=32):
        z = self.ae.encode(x, batch_size)
        y = self.ae.decode(z, batch_size)
        return [y]

from config import get_config
from utils import prepare_dirs_and_logger
from keras_data import BatchManager, copy_dataset_info
import os
from utils import save_image
from LatentSpacePhysics.src.util.requirements import init_packages
init_packages()

import git

#---------------------------------------------------------------------------------
if __name__ == "__main__":
    config, unparsed = get_config()
    prepare_dirs_and_logger(config)

    # create GIT file
    repo = git.Repo(search_parent_directories=False)
    open("{}/{}".format(config.model_dir, repo.head.object.hexsha), "w") 

    # copy dataset info to model dir
    copy_dataset_info(config)

    # Transfer data to local vars
    batch_num = config.batch_size
    validation_split = 0.1
    epochs = config.epochs
    input_frame_count = config.input_frame_count
    prediction_window = config.w_num
    decode_predictions = config.decode_predictions
    skip_pred_steps = config.skip_pred_steps
    init_state_network = config.init_state_network
    in_out_states = config.in_out_states
    pred_gradient_loss = config.pred_gradient_loss
    ls_prediction_loss = config.ls_prediction_loss
    ls_supervision = config.ls_supervision
    sqrd_diff_loss = config.sqrd_diff_loss
    ls_split = config.ls_split
    test_data_types = config.data_type.copy()
    if "inflow" in test_data_types: test_data_types.remove("inflow")

    # Multitile model
    config.tile_scale = 3 # => 3x3 tiles

    keras_batch_manager = BatchManager(config, input_frame_count, prediction_window)
    sup_param_count = keras_batch_manager.supervised_param_count

    in_out_dim = 3 if "density" in config.data_type or "levelset" in config.data_type else 2
    in_out_dim = in_out_dim + 1 if config.is_3d else in_out_dim
    input_shape = (input_frame_count-1,9,)
    input_shape += (config.res_z,) if config.is_3d else ()
    input_shape += (config.res_y, config.res_x, in_out_dim)

    train_prediction_only = config.train_prediction_only and config.is_train and config.load_path is not ''
    if train_prediction_only: 
        print("Training only the prediction network!")

    # Write config to file
    config_d = vars(config) if config else {}
    unparsed_d = vars(unparsed) if unparsed else {}

    with open(config.model_dir + "/input_args.json", 'w') as fp:
        json.dump({**config_d, **unparsed_d}, fp)

    print("Input Shape: {}".format(input_shape))

    rec_pred = RecursivePrediction_Multitile(config=config, input_shape=input_shape, decode_predictions=decode_predictions, skip_pred_steps=skip_pred_steps, init_state_network=init_state_network, in_out_states=in_out_states, pred_gradient_loss=pred_gradient_loss, ls_prediction_loss=ls_prediction_loss, ls_supervision=ls_supervision, sqrd_diff_loss=sqrd_diff_loss, ls_split=ls_split, supervised_parameters=sup_param_count, train_prediction_only=train_prediction_only) 

    # Train =====================================================================================================
    if config.is_train:
        if config.load_path:
            rec_pred.load_model(config.load_path, load_ae=True, load_pred=False)
            if train_prediction_only:
                rec_pred.pred.model.trainable = True
                rec_pred.ae._encoder.trainable = False
                rec_pred.ae._decoder.trainable = False
            
            rec_pred.pred._compile_model()
            rec_pred.ae._compile_model()
            rec_pred._compile_model()

        test_data = keras_batch_manager.batch_with_name(min(batch_num,8), validation_split=validation_split, validation=True, adjust_to_batch=True, data_types=test_data_types, use_tiles=keras_batch_manager.tile_generator is not None)
        test_data = np.array(next(test_data)[0])
        print("test_data shape: {}".format(test_data.shape))
        test_data = test_data[:,int(test_data.shape[1]/3):-int(test_data.shape[1]/3), int(test_data.shape[2]/3):-int(test_data.shape[2]/3)]
        print("test_data shape: {}".format(test_data.shape))

        # Input
        # (8, 2, 9, 8, 8, 3)
        # Output
        # (8, 8, 8, 3)
        # (8, 8, 8, 1)
        # (8, 24, 24, 3)
        test_generator = keras_batch_manager.generator_ae_tile_sequence(min(batch_num,8), validation_split=validation_split, validation=True, ls_split_loss=ls_split > 0.0, advection_loss=config.advection_loss > 0.0)
        test_data_input, test_data_output = next(test_generator)
        print("TEST DATA")
        test_data_input = test_data_input[0]
        print(test_data_input.shape)
        for a in test_data_output:
            print(a.shape)
        print("~TEST DATA")

        if keras_batch_manager.is_3d:
            save_img_to_disk_3d(test_data_output[0], 0, config.model_dir, keras_batch_manager, "x_fixed_gt")
            plot_callback = PlotPredFields(rec_pred.predict, test_data_input, save_img_to_disk_3d, config.model_dir, keras_batch_manager)
        else:
            save_img_to_disk(test_data_output[0], 0, config.model_dir, keras_batch_manager, "x_fixed_gt_vdt1")
            save_img_to_disk(test_data_output[1], 0, config.model_dir, keras_batch_manager, "x_fixed_gt_dt2")
            plot_callback = PlotPredFields(rec_pred.predict, test_data_input, save_img_to_disk, config.model_dir, keras_batch_manager)

        hist = rec_pred.train(epochs, batch_manager=keras_batch_manager, batch_size=batch_num, validation_split=validation_split, callbacks=[plot_callback])
        rec_pred.save_model(config.model_dir)

        import LatentSpacePhysics.src.util.plot as plot
        import json
        if hist is not None:
            with open(config.model_dir+"/combined_hist.json", 'w') as f:
                json.dump(hist.history, f, indent=4)

        # plot the history
        if hist:
            lstm_history_plotter = plot.Plotter()
            lstm_history_plotter.plot_history(hist.history)
            lstm_history_plotter.save_figures(config.model_dir+"/", "Combined_History", filetype="svg")
            lstm_history_plotter.save_figures(config.model_dir+"/", "Combined_History", filetype="png")

    # Test =====================================================================================================
    else:
        # ===============================================================================================
        # Check AE encdec after load
        rec_pred = RecursivePrediction_Multitile(config=config, input_shape=input_shape, decode_predictions=decode_predictions, skip_pred_steps=skip_pred_steps, init_state_network=init_state_network, in_out_states=in_out_states, pred_gradient_loss=pred_gradient_loss, ls_prediction_loss=ls_prediction_loss, ls_supervision=ls_supervision, sqrd_diff_loss=sqrd_diff_loss, ls_split=ls_split, supervised_parameters=sup_param_count) 

        rec_pred.load_model("folder to dir") # load_path argument
        ae = rec_pred.ae
        test_data = keras_batch_manager.batch_with_name(1, validation_split=1, validation=True, randomized=False, data_types=test_data_types)
        i = 0

        for x, path, sup_param in test_data:
            print(config.model_dir)
            x = np.array(x)
            save_img_to_disk_3d(x*2, 0, config.model_dir, keras_batch_manager, "x_gt_{}".format(i))
            gt_vel = x[0][...,:3]
            print(gt_vel.shape)
            np.savez_compressed(config.model_dir + "/v_gt_{}.npz".format(i), arr_0=gt_vel, rot=sup_param[0][0], pos=sup_param[0][1])

            enc_x = ae.encode(x, batch_size=1)
            enc_x[0, -2] = sup_param[0][0]
            enc_x[0, -1] = sup_param[0][1]
            dec_x = ae.decode(enc_x, batch_size=1)
            save_img_to_disk_3d(dec_x*2, 0, config.model_dir, keras_batch_manager, "x_encdec_{}".format(i))
            encdec_vel = dec_x[0][...,:3]
            print(encdec_vel.shape)
            np.savez_compressed(config.model_dir + "/v_encdec_{}.npz".format(i), arr_0=encdec_vel, rot=sup_param[0][0], pos=sup_param[0][1])
            i = i+1
