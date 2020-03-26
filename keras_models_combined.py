import time

import tensorflow as tf

from keras_models_general import *
from keras_models_spatial import *
from keras_models_temporal import *

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


#=====================================================================================
class RecursivePrediction(Network):
    #---------------------------------------------------------------------------------
    def _init_vars(self, config, settings=None, **kwargs):
        self.input_shape = kwargs.get("input_shape", (4, 64, 64, 64, 3) if config.is_3d else (4, 128, 96, 2))
        
        self.tf_run_options = None

        # Submodel Vars
        self.ae = None
        self.pred = None

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
        if hasattr(config, 'only_last_prediction'):
            self.only_last_prediction = config.only_last_prediction
        else:
            self.only_last_prediction = False
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
        if self.train_prediction_only:
            loss_list = []
        else:
            loss_list = [ AE_Loss(sqrd_diff_loss=self.sqrd_diff_loss, density=use_density, is_3d=self.is_3d) ]
            self.loss_weights.append(1.0)
        if self.decode_predictions:
            loss_list.append(Pred_Decoded_Loss(skip_steps=self.skip_pred_steps, gradient_loss=self.pred_gradient_loss, sqrd_diff_loss=self.sqrd_diff_loss, density=use_density, is_3d=self.is_3d))
            self.loss_weights.append(1.0)
        else:
            loss_list.append(Pred_Loss(1 if self.only_last_prediction else self.recursive_prediction, skip_steps=self.skip_pred_steps, gradient_loss=self.pred_gradient_loss, sqrd_diff_loss=self.sqrd_diff_loss, density=use_density, is_3d=self.is_3d))
            self.loss_weights.append(1.0)
        if not self.train_prediction_only:
            loss_list.append("mse")
            self.loss_weights.append(1.0)
        if self.ls_prediction_loss:
            assert False, "ls_prediction_loss is currently unsupported"
            loss_list.append(Pred_Loss(1 if self.only_last_prediction else self.recursive_prediction, skip_steps=self.skip_pred_steps, gradient_loss=False))
            self.loss_weights.append(1.0)
        if self.ls_split > 0.0:
            self.ls_split_idx = int(self.z_num * self.ls_split)
            assert self.ls_split_idx > 0, "ls_split_idx must be larger than 0!"
            if not self.train_prediction_only:
                loss_list.append(Split_Loss(self.ls_split_idx, self.z_num - self.sup_param_count)) # one is skipped for supervised parameter
                loss_list.append(Split_Loss(0, self.ls_split_idx))
                self.loss_weights.append(1.0)
                self.loss_weights.append(1.0)
            print("Splitting LS at  0 -> {} and {} -> {}".format(self.ls_split_idx, self.ls_split_idx, self.z_num - self.sup_param_count))
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
            self.pred = Prediction(config=self.config, input_shape=(self.w_num, self.z_num), stateful=self.stateful, in_out_states=self.in_out_states) # (b, 2, 32)
            self.pred._build_model()

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

        # Load predefined model layouts
        self._create_submodels()

        enc = self.ae._encoder
        dec = self.ae._decoder
        p_pred = self.ae._p_pred 
        pred = self.pred.model

        # State init network
        if self.in_out_states:
            state_init_in = Input(shape=(self.w_num, self.z_num), dtype="float32", name="State_Init_Input")
            state_init_x = Reshape((self.w_num * self.z_num,), name="Reshape_io_states")(state_init_in)
            state_init_x = Dense(128)(state_init_x)
            state_init_x = LeakyReLU()(state_init_x)
            state_init_x = Dense(2*self.pred.encoder_lstm_neurons + 2*self.pred.decoder_lstm_neurons)(state_init_x)
            state_init_x = LeakyReLU()(state_init_x)
            state_init_x = Reshape((4, self.pred.encoder_lstm_neurons), name="Reshape_io_states_2")(state_init_x)
            self.state_init_model = Model(name="State_Init", inputs=state_init_in, outputs=state_init_x)

        if self.stateful:
            inputs = Input(batch_shape=(self.b_num,) + self.input_shape, dtype="float32", name="Combined_AE_Input_Fields") # (b, input_depth, x, y, c)
        else:
            inputs = Input(shape=self.input_shape, dtype="float32", name="Combined_AE_Input_Fields") # (b, input_depth, y, x, c)

        # Input for GT supervised parameters (e.g. rotation and position)
        # -> (b_num, 14, 2)
        sup_param_inputs = Input(shape=(self.input_shape[0], self.sup_param_count), dtype="float32", name="Combined_AE_Input_Sup_Param")

        if self.use_inflow or self.advection_loss > 0.0:
            input_inflow = Input(shape=self.input_inflow_shape, dtype="float32", name="Inflow_Input") # (b, input_depth, y, x, 1)

        if self.ls_split > 0.0 and not self.train_prediction_only:
            inputs_full = Lambda(lambda x: x[:, 0:1], name="ls_split_slice")(inputs)
            inputs_full = Lambda(lambda x: K.squeeze(x,1), name="ls_split_0")(inputs_full)
            inputs_vel = Lambda(lambda x: K.concatenate([x[...,0:velo_dim], K.zeros_like(x)[...,velo_dim:velo_dim+1]], axis=-1), name="ls_split_1")(inputs_full)
            inputs_den = Lambda(lambda x: K.concatenate([K.zeros_like(x)[...,0:velo_dim], x[...,velo_dim:velo_dim+1]], axis=-1), name="ls_split_2")(inputs_full)
            z_vel = enc(inputs_vel)
            z_vel = Lambda(lambda x: x, name="z_vel")(z_vel)
            z_den = enc(inputs_den)
            z_den = Lambda(lambda x: x, name="z_den")(z_den)

        enc_input = None
        enc_input_range = inputs.shape[1] if self.ls_prediction_loss else self.w_num
        for i in range(enc_input_range): # input depth iteration
            if enc_input == None:
                enc_input = Lambda(lambda x: x[:, i], name="Slice_enc_input_{}".format(i))(inputs)
                enc_input = enc(enc_input)
                enc_input = Lambda(lambda x: K.expand_dims(x, axis=1))(enc_input)
            else:
                temp_enc = Lambda(lambda x: x[:, i], name="Slice_enc_input_{}".format(i))(inputs)
                temp_enc = enc(temp_enc)
                encoded = Lambda(lambda x: K.expand_dims(x, axis=1))(temp_enc)
                enc_input = concatenate([enc_input, encoded], axis=1) # (b, input_depth, z)

        # directly extract z to apply supervised latent space loss afterwards
        z = Lambda(lambda x: x[:, 0:1], name="Slice_z")(enc_input)

        # Overwrite supervised latent space entries in enc_input
        # e.g. sup_param_inputs -> (b,14,2)
        enc_input = Lambda(lambda x: x[:, :, 0:-self.sup_param_count], name="sup_param_count_slice")(enc_input)

        # (b_num, 14, 2) -> (b_num, w_num, 2)
        first_input_sup_params = Lambda(lambda x: x[:, :self.w_num], name="first_input_sup_param_slice")(sup_param_inputs)
        enc_input = concatenate([enc_input, first_input_sup_params], axis=2, name="enc_input_sup_param_concat")

        rec_input = enc_input

        if self.in_out_states:
            if self.init_state_network:
                pred_states_init = self.state_init_model(rec_input)
            else:
                pred_states_init = Lambda(lambda x: K.zeros( (self.b_num, 4, self.pred.encoder_lstm_neurons) ))(inputs) # lambda is quickhack to make initializing with zero possible (input tensor does not really matter)...

            def slice_states(x):
                return tf.unstack(x, axis=1)
            pred_states_0_0, pred_states_0_1, pred_states_1_0, pred_states_1_1 = Lambda(slice_states)(pred_states_init)

        if self.ls_prediction_loss:
            rec_output_ls = None
        rec_output = None
        adv_output = None
        rec_den = None
        for i in range(self.recursive_prediction):
            if self.in_out_states:
                x, pred_states_0_0, pred_states_0_1, pred_states_1_0, pred_states_1_1 = pred([rec_input, pred_states_0_0, pred_states_0_1, pred_states_1_0, pred_states_1_1])
            else:
                x = pred([rec_input])
            x = self.pred._fix_output_dimension(x)

            # predicted delta 
            # add now to previous input
            pred_add_first_elem = Lambda(lambda x: x[:, -self.pred.out_w_num:None], name="rec_input_add_slice_{}".format(i))(rec_input)
            x = Add(name="Pred_Add_{}".format(i))([pred_add_first_elem, x]) # previous z + predicted delta z

            if self.ls_supervision:
                pred_x = Lambda(lambda x: x[:, :, 0:-self.sup_param_count], name="pred_x_slice_{}".format(i))(x)

                sup_param_real = Lambda(lambda x: x[:, self.w_num+i:self.w_num+self.pred.out_w_num+i], name="sup_param_real_{}".format(i))(sup_param_inputs)
                x = concatenate([pred_x, sup_param_real], axis=2, name="Pred_Real_Supervised_Concat_{}".format(i))

            rec_input = Lambda(lambda x: x[:, self.pred.out_w_num:None], name="rec_input_slice_{}".format(i))(rec_input)

            rec_input = concatenate([rec_input, x], axis=1, name="Pred_Input_Concat_{}".format(i))
            rec_input_last = x
            if self.decode_predictions:
                if self.ls_prediction_loss:
                    x_ls = x
                x = dec(Reshape((self.z_num,), name="Reshape_xDecPred_{}".format(i))(x))

            # ########################################################################################################################
            # density/ls advection loss
            # 0) get first GT density field that is to be advected (0,1) -> 2 [take 1]
            # 0) denormalize current passive GT field (z,y,x,1)
            # 1) extract velocity array (z,y,x,3) [or (...,2)]
            # 2) denormalize velocity -> v = keras_data.denorm_vel(v)
            # 3) apply inflow region or obstacle subtract
            # 4) use current passive field (z,y,x,1) as advection src
            # 5) call advect(src, v, dt=keras_data.time_step, mac_adv=False, name="density")
            # 6) store as d+1 for usage in next frame -> rec_den
            # 7) normalize returned advected passive quantity
            # 8) hand over to loss -> (advect(d^t,v^t), d^t+1)
            # 9) use the advected density for reencoding 
            # 10) start at 1)

            if self.advection_loss > 0.0 and i < self.recursive_prediction - 1:
                assert self.decode_predictions, ("decode_predictions must be used")
                cur_decoded_pred = x
                # 0) get first GT density field that is to be advected (0,1) -> 2 [take 1]
                if rec_den == None:
                    rec_den = Lambda(lambda x: x[:, self.w_num-1, ..., velo_dim:velo_dim+1], name="gt_passive_{}".format(i))(inputs)
                    rec_den = batch_manager.denorm(rec_den, self.passive_data_type, as_layer=True)

                # 1) extract velocity array (z,y,x,3) [or (...,2)]
                pred_vel = Lambda(lambda x: x[...,0:velo_dim], name="vel_extract_{}".format(i))(cur_decoded_pred)
                # 2) denormalize velocity -> v = keras_data.denorm_vel(v)
                denorm_pred_vel = batch_manager.denorm(pred_vel, "velocity", as_layer=True)
                # 3) apply inflow region or obstacle subtract
                cur_inflow = Lambda(lambda x: x[:, self.w_num+i], name="inflow_extract_{}".format(self.w_num+i))(input_inflow) 
                rec_den = Lambda(lambda x: K.tf.where(tf.greater(x[0], 0.0), x[0], x[1]))([cur_inflow, rec_den])
                # 4) use current passive field (z,y,x,1) as advection src
                # 5) call advect(src, v, dt=keras_data.time_step, mac_adv=False, name="density")
                # 6) store as d+1 for usage in next frame -> rec_den
                #print("4) + 5) + 6)")
                rec_den = Lambda(advect, arguments={'dt': batch_manager.time_step, 'mac_adv': False, 'name': self.passive_data_type})([rec_den, denorm_pred_vel])
                # 7) normalize returned advected passive quantity
                rec_den_norm = batch_manager.norm(rec_den, self.passive_data_type, as_layer=True)
                # 8) hand over to loss -> (advect(d^t,v^t), d^t+1)
                if adv_output == None or self.only_last_prediction:
                    rec_den_norm = Lambda(lambda x: K.expand_dims(x, axis=1))(rec_den_norm)
                    adv_output = rec_den_norm
                else:
                    rec_den_norm = Lambda(lambda x: K.expand_dims(x, axis=1))(rec_den_norm)
                    adv_output = concatenate([adv_output, rec_den_norm], axis=1, name="Adv_Passive_GT_Concat_{}".format(i))
                # 9) use the advected density for reencoding
                rec_den_norm_sq = Lambda(lambda x: K.squeeze(x,1), name="rec_den_norm_squeeze_{}".format(i))(rec_den_norm)
                reencoded_input = Lambda(lambda x: K.concatenate(x, axis=-1), name="reencoding_vel_den_{}".format(self.w_num+i))([pred_vel, rec_den_norm_sq])
                z_reenc = enc(reencoded_input)
                z_reenc = Lambda(lambda x: K.expand_dims(x, axis=1))(z_reenc)
                # 10) take only density part of latent space and replace ls history
                # create mask with npa = np.zeros(shape); npa[:, x:y] = 1; m = K.constant( npa )
                m_np = np.zeros( (self.pred.out_w_num, self.z_num), dtype=np.float32)
                m_np[:,self.ls_split_idx:-self.sup_param_count] = 1.0
                # create lambda with a,b: a * m + b * (1-m)
                rec_input_last = Lambda(lambda x: x[0] * K.constant(value=m_np, dtype='float32') + x[1] * (1.0-K.constant(value=m_np, dtype='float32')), name="z_reenc_stitch_{}".format(self.w_num + i))( [z_reenc, rec_input_last] )
                # replace rec_input last elem
                rec_input = Lambda(lambda x: x[:, :-1], name="rec_input_cut_{}".format(self.w_num+i))(rec_input) 
                rec_input = concatenate([rec_input, rec_input_last], axis=1, name="rec_input_concat_{}".format(self.w_num+i))

            if rec_output == None or self.only_last_prediction:
                rec_output = x
            else:
                rec_output = concatenate([rec_output, x], axis=1, name="Pred_Output_Concat_{}".format(i))

            if self.ls_prediction_loss:
                if rec_output_ls == None or self.only_last_prediction:
                    rec_output_ls = x_ls
                else:
                    rec_output_ls = concatenate([rec_output_ls, x_ls], axis=1, name="Pred_Output_LS_Concat_{}".format(i))

        if self.decode_predictions:
            if self.only_last_prediction:
                rec_out_shape = (1,)+self.input_shape[1:]
            else:
                rec_out_shape = (self.recursive_prediction,)+self.input_shape[1:]
            rec_output = Reshape(rec_out_shape, name="Prediction_output")(rec_output)

        if self.decode_predictions:
            if self.ls_prediction_loss:
                if self.only_last_prediction:
                    GT_output_LS = Lambda(lambda x: x[:, -1], name="GT_output_LS_slice".format(i))(enc_input)
                    GT_output_LS_shape = (1,)+int_shape(GT_output_LS)[1:]
                    GT_output_LS = Reshape(GT_output_LS_shape, name="Reshape_last_GT_ls")(GT_output_LS)
                else:
                    GT_output_LS = Lambda(lambda x: x[:, -self.recursive_prediction:None], name="GT_output_LS_slice".format(i))(enc_input)
        else:
            if self.only_last_prediction:
                GT_output = Lambda(lambda x: x[:, -1], name="GT_output_encoded_slice".format(i))(enc_input)
                GT_output_shape = (1,)+int_shape(GT_output)[1:]
                GT_output = Reshape(GT_output_shape, name="Reshape_last_GT")(GT_output)
            else:
                GT_output = Lambda(lambda x: x[:, -self.recursive_prediction:None], name="GT_output_encoded_slice".format(i))(enc_input)

        # first half of pred_output is actual prediction, last half is GT to compare against in loss
        if not self.decode_predictions:
            pred_output = concatenate([rec_output, GT_output], axis=1, name="Prediction_Output")
        else:
            pred_output = rec_output

        if self.decode_predictions and self.ls_prediction_loss:
            pred_output_LS = concatenate([rec_output_ls, GT_output_LS], axis=1, name="Prediction_Output_LS")

        # supervised LS loss
        p_pred_output = p_pred(Reshape((self.z_num,), name="Reshape_pPred")(z))
        
        # decoder loss 
        if not self.train_prediction_only:
            ae_output = dec(Reshape((self.z_num,), name="Reshape_zTrainPredOnly")(z))
            output_list = [ae_output, pred_output]
        else:
            output_list = [pred_output]
        
        if not self.train_prediction_only:
            output_list.append(p_pred_output)
        if self.ls_prediction_loss:
            output_list.append(pred_output_LS)
        if self.ls_split > 0.0 and not self.train_prediction_only:
            output_list.append(z_vel)
            output_list.append(z_den)
        if self.advection_loss > 0.0:
            output_list.append(adv_output)

        input_list = [inputs, sup_param_inputs]
        if self.use_inflow or self.advection_loss > 0.0:
            input_list.append(input_inflow)

        print("Setup Model")
        if len(self.gpus) > 1:
            with tf.device('/cpu:0'):
                self.model = Model(name="Combined_AE_LSTM", inputs=input_list, outputs=output_list)
        else:
            self.model = Model(name="Combined_AE_LSTM", inputs=input_list, outputs=output_list)

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
            train_generator = batch_manager.generator_ae_sequence(batch_size, validation_split, validation=False, decode_predictions=self.decode_predictions, ls_prediction_loss=self.ls_prediction_loss, ls_split_loss=self.ls_split > 0.0, train_prediction_only=self.train_prediction_only, advection_loss=self.advection_loss > 0.0)

            # validation samples
            val_gen_nb_samples = batch_manager.steps_per_epoch(batch_size, validation_split, validation=True)
            assert val_gen_nb_samples > 0, ("Batch size is too large for current scene samples/timestep settings. Training by generator not possible. Please adjust the batch size in the 'settings.json' file.")
            print ("Number of validation batch samples per epoch: {}".format(val_gen_nb_samples))
            validation_generator = batch_manager.generator_ae_sequence(batch_size, validation_split, validation=True, decode_predictions=self.decode_predictions, ls_prediction_loss=self.ls_prediction_loss, ls_split_loss=self.ls_split > 0.0, train_prediction_only=self.train_prediction_only, advection_loss=self.advection_loss > 0.0)

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
        if self.in_out_states:
            model_to_json(self.state_init_model, path + "/state_init.json")
            save_model(self.state_init_model, path + "/state_init.h5")

    #---------------------------------------------------------------------------------
    def predict(self, x, batch_size=32):
        return self.model.predict(x, batch_size=batch_size)

    #---------------------------------------------------------------------------------
    def ae_predict(self, x, batch_size=32):
        return self.ae.predict(x, batch_size=batch_size)


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

    keras_batch_manager = BatchManager(config, input_frame_count, prediction_window)
    sup_param_count = keras_batch_manager.supervised_param_count

    in_out_dim = 3 if "density" in config.data_type or "levelset" in config.data_type else 2
    in_out_dim = in_out_dim + 1 if config.is_3d else in_out_dim
    input_shape = (input_frame_count,)
    input_shape += (config.res_z,) if config.is_3d else ()
    input_shape += (config.res_y, config.res_x, in_out_dim)

    train_prediction_only = config.train_prediction_only and config.is_train and config.load_path is not ''
    if train_prediction_only: 
        print("Training only the prediction network!")

    ## Write config to file
    config_d = vars(config) if config else {}
    unparsed_d = vars(unparsed) if unparsed else {}

    with open(config.model_dir + "/input_args.json", 'w') as fp:
        json.dump({**config_d, **unparsed_d}, fp)

    print("Input Shape: {}".format(input_shape))

    rec_pred = RecursivePrediction(config=config, input_shape=input_shape, decode_predictions=decode_predictions, skip_pred_steps=skip_pred_steps, init_state_network=init_state_network, in_out_states=in_out_states, pred_gradient_loss=pred_gradient_loss, ls_prediction_loss=ls_prediction_loss, ls_supervision=ls_supervision, sqrd_diff_loss=sqrd_diff_loss, ls_split=ls_split, supervised_parameters=sup_param_count, train_prediction_only=train_prediction_only) 

    # Train =====================================================================================================
    if config.is_train:
        if config.load_path:
            rec_pred.load_model(config.load_path, load_ae=config.load_ae, load_pred=config.load_pred)
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
        if keras_batch_manager.is_3d:
            save_img_to_disk_3d(test_data, 0, config.model_dir, keras_batch_manager, "x_fixed_gt")
            plot_callback = PlotAEFields(rec_pred.ae_predict, test_data, save_img_to_disk_3d, config.model_dir, keras_batch_manager)
        else:
            save_img_to_disk(test_data, 0, config.model_dir, keras_batch_manager, "x_fixed_gt")
            plot_callback = PlotAEFields(rec_pred.ae_predict, test_data, save_img_to_disk, config.model_dir, keras_batch_manager)

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
        rec_pred = RecursivePrediction(config=config, input_shape=input_shape, decode_predictions=decode_predictions, skip_pred_steps=skip_pred_steps, init_state_network=init_state_network, in_out_states=in_out_states, pred_gradient_loss=pred_gradient_loss, ls_prediction_loss=ls_prediction_loss, ls_supervision=ls_supervision, sqrd_diff_loss=sqrd_diff_loss, ls_split=ls_split, supervised_parameters=sup_param_count) 

        rec_pred.load_model("./path_to_network/") # load_path argument

        ae = rec_pred.ae
        test_data = keras_batch_manager.batch_with_name(1, validation_split=1, validation=True, randomized=False, data_types=test_data_types)
        i = 0

        for x, path, sup_param in test_data:
            x = np.array(x)
            save_img_to_disk_3d(x*2, 0, config.model_dir, keras_batch_manager, "x_gt_{}".format(i))
            gt_vel = x[0][...,:3]
            np.savez_compressed(config.model_dir + "/v_gt_{}.npz".format(i), arr_0=gt_vel, rot=sup_param[0][0], pos=sup_param[0][1])

            enc_x = ae.encode(x, batch_size=1)
            enc_x[0, -2] = sup_param[0][0]
            enc_x[0, -1] = sup_param[0][1]
            dec_x = ae.decode(enc_x, batch_size=1)
            save_img_to_disk_3d(dec_x*2, 0, config.model_dir, keras_batch_manager, "x_encdec_{}".format(i))
            encdec_vel = dec_x[0][...,:3]
            np.savez_compressed(config.model_dir + "/v_encdec_{}.npz".format(i), arr_0=encdec_vel, rot=sup_param[0][0], pos=sup_param[0][1])
            i = i+1
