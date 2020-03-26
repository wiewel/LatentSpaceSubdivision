from keras_models_combined import *

#=====================================================================================
class RecursivePredictionCleanSplit(Network):
    #---------------------------------------------------------------------------------
    def _init_vars(self, config, settings=None, **kwargs):
        self.input_shape = kwargs.get("input_shape", (4, 64, 64, 64, 3) if config.is_3d else (4, 128, 96, 2))
        
        # Submodel Vars
        self.ae_v = None
        self.ae_d = None
        self._p_pred_d = None
        self._p_pred_v = None
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
        assert self.stateful == False, "stateful is not supported"

        self.decode_predictions = kwargs.get("decode_predictions", False)
        self.skip_pred_steps = kwargs.get("skip_pred_steps", False)
        self.init_state_network = kwargs.get("init_state_network", False)
        self.pred_gradient_loss = kwargs.get("pred_gradient_loss", False)
        self.ls_prediction_loss = kwargs.get("ls_prediction_loss", False)
        self.ls_supervision = kwargs.get("ls_supervision", False)
        self.sqrd_diff_loss = kwargs.get("sqrd_diff_loss", False)
        self.ls_split = kwargs.get("ls_split", 0.0)
        self.sup_param_count = kwargs.get("supervised_parameters", 1)

        # TRAINER VARS
        self.config = config
        self.kwargs = kwargs

        self.arch = config.arch
        self.is_3d = config.is_3d

        self.optimizer = config.optimizer
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        self.model_dir = config.model_dir
        self.load_path = config.load_path
        self.nn_path = config.nn_path

        self.is_train = config.is_train

        self.dataset = config.dataset

        self.b_num = config.batch_size
        self.z_num = config.z_num
        self.w_num = config.w_num

        self.recursive_prediction = self.input_shape[0] - self.w_num
        print("self.recursive_prediction {}".format(self.recursive_prediction))

        use_density = "density" in self.config.data_type

        loss_list = [
            AE_Loss(sqrd_diff_loss=self.sqrd_diff_loss, density=use_density, is_3d=self.is_3d),
            Pred_Loss(self.recursive_prediction, skip_steps=self.skip_pred_steps, gradient_loss=self.pred_gradient_loss, sqrd_diff_loss=self.sqrd_diff_loss, density=use_density, is_3d=self.is_3d),
            "mse",
            "mse"
        ]

        if self.ls_split > 0.0:
            self.z_num_vel = int(self.z_num * self.ls_split)
            self.z_num_den = self.z_num - self.z_num_vel
            assert self.z_num_vel + self.z_num_den == self.z_num, "total z_num is not summing up"

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
    import copy
    def _create_submodels(self):
        if not self.ae_v:
            kwargs_copy = copy.deepcopy(self.kwargs)
            print("Velo AE")
            print(kwargs_copy["input_shape"])
            in_v = list(kwargs_copy["input_shape"])
            in_v[-1] = in_v[-1] - 1
            in_v = in_v[1:]
            kwargs_copy["input_shape"] = tuple(in_v)
            print(kwargs_copy["input_shape"])
            kwargs_copy["name_prefix"] = "AE_v"
            config_copy = copy.deepcopy(self.config)
            config_copy.data_type = "velocity"
            config_copy.z_num = self.z_num_vel
            self.ae_v = Autoencoder(config=config_copy, **kwargs_copy)
            self.ae_v._build_model()
        if not self.ae_d:
            kwargs_copy = copy.deepcopy(self.kwargs)
            print("Den AE")
            print(kwargs_copy["input_shape"])
            in_d = list(kwargs_copy["input_shape"])
            in_d[-1] = 1
            in_d = in_d[1:]
            kwargs_copy["input_shape"] = tuple(in_d)
            print(kwargs_copy["input_shape"])
            kwargs_copy["name_prefix"] = "AE_d"
            config_copy = copy.deepcopy(self.config)
            config_copy.data_type = "density"
            config_copy.z_num = self.z_num_den
            self.ae_d = Autoencoder(config=config_copy, **kwargs_copy)
            self.ae_d._build_model()
        if not self._p_pred_v:
            pred_input = Input(shape=(self.z_num_vel,))
            p_pred_out = Lambda(lambda x: x[:, -self.sup_param_count:], name="p_vel")(pred_input)
            self._p_pred_v = Model(name="p_Pred_v", inputs=pred_input, outputs=p_pred_out)
        if not self._p_pred_d:
            pred_input = Input(shape=(self.z_num_den,))
            p_pred_out = Lambda(lambda x: x[:, -self.sup_param_count:], name="p_den")(pred_input)
            self._p_pred_d = Model(name="p_Pred_d", inputs=pred_input, outputs=p_pred_out)
        if not self.pred:
            print(self.z_num)
            print(self.config)

            self.pred = Prediction(config=self.config, input_shape=(self.w_num, self.z_num), stateful=self.stateful, in_out_states=False) # (b, 2, 32)
            self.pred._build_model()
            self.pred.model.summary()

    #---------------------------------------------------------------------------------
    def _build_model(self):
        print("Building Model")

        velo_dim = 3 if self.is_3d else 2

        self._create_submodels()

        enc_v = self.ae_v._encoder
        dec_v = self.ae_v._decoder
        p_pred_v = self._p_pred_v
        enc_d = self.ae_d._encoder
        dec_d = self.ae_d._decoder
        p_pred_d = self._p_pred_d 

        pred = self.pred.model

        inputs = Input(shape=self.input_shape, dtype="float32", name="Combined_AE_Input_Fields") # (b, input_depth, x, y, c)

        # Input for GT supervised parameters (e.g. rotation and position)
        # -> (b_num, 14, 2)
        sup_param_inputs = Input(shape=(self.input_shape[0], self.sup_param_count), dtype="float32", name="Combined_AE_Input_Sup_Param")

        enc_input = None
        for i in range(inputs.shape[1]): # input depth iteration
            if enc_input == None:
                enc_input = Lambda(lambda x: x[:, i:i+1], name="Slice_enc_input_{}".format(i))(inputs)
                enc_input = Reshape(enc_input.shape[2:], name="Reshape_enc_input_{}".format(i))(enc_input)
                
                enc_input_v = Lambda(lambda x: x[...,0:velo_dim], name="Vel_Input_{}".format(i))(enc_input)
                enc_input_d = Lambda(lambda x: x[...,velo_dim:velo_dim+1], name="Den_Input_{}".format(i))(enc_input)
                print("enc_input_v {}".format(enc_input_v.shape))
                print("enc_input_d {}".format(enc_input_d.shape))
                z_vel_first = enc_v(enc_input_v)
                z_den_first = enc_d(enc_input_d)

                # Overwrite supervised latent space entries in enc_input
                z_vel_first_sp = Lambda(lambda x: x[:, 0:-self.sup_param_count], name="sup_param_count_slice_v_{}".format(i))(z_vel_first)
                z_den_first_sp = Lambda(lambda x: x[:, 0:-self.sup_param_count], name="sup_param_count_slice_d_{}".format(i))(z_den_first)

                sp_slice = Lambda(lambda x: x[:, i], name="sp_slice_{}".format(i))(sup_param_inputs)
                print("SP_SLICE: {}".format(sp_slice.shape))

                z_vel_first_sp = concatenate([z_vel_first_sp, sp_slice], axis=1, name="z_vel_concat_{}".format(i))
                print("z_vel_first_sp with supervised params {}".format(z_vel_first_sp.shape))
                z_den_first_sp = concatenate([z_den_first_sp, sp_slice], axis=1, name="z_den_concat_{}".format(i))
                print("z_den_first_sp with supervised params {}".format(z_den_first_sp.shape))

                enc_input = concatenate([z_vel_first_sp, z_den_first_sp], axis=-1)
                print("enc_input {}".format(enc_input.shape))

                enc_input = Lambda(lambda x: K.expand_dims(x, axis=1))(enc_input)

                # invalidate temp vars
                z_vel_first_sp = None
                z_den_first_sp = None
                sp_slice = None
                enc_input_v = None
                enc_input_d = None
            else:
                temp_enc = Lambda(lambda x: x[:, i:i+1], name="Slice_enc_input_{}".format(i))(inputs)
                temp_enc = Reshape(temp_enc.shape[2:], name="Reshape_enc_input_{}".format(i))(temp_enc)

                enc_input_v = Lambda(lambda x: x[...,0:velo_dim], name="Vel_Input_{}".format(i))(temp_enc)
                enc_input_d = Lambda(lambda x: x[...,velo_dim:velo_dim+1], name="Den_Input_{}".format(i))(temp_enc)
                print("enc_input_v {}".format(enc_input_v.shape))
                print("enc_input_d {}".format(enc_input_d.shape))
                z_vel = enc_v(enc_input_v)
                z_den = enc_d(enc_input_d)

                # Overwrite supervised latent space entries in enc_input
                z_vel = Lambda(lambda x: x[:, 0:-self.sup_param_count], name="sup_param_count_slice_v_{}".format(i))(z_vel)
                z_den = Lambda(lambda x: x[:, 0:-self.sup_param_count], name="sup_param_count_slice_d_{}".format(i))(z_den)

                sp_slice = Lambda(lambda x: x[:, i], name="sp_slice_{}".format(i))(sup_param_inputs)
                print("SP_SLICE: {}".format(sp_slice.shape))

                z_vel = concatenate([z_vel, sp_slice], axis=1, name="z_vel_concat_{}".format(i))
                print("z_vel with supervised params {}".format(z_vel.shape))
                z_den = concatenate([z_den, sp_slice], axis=1, name="z_den_concat_{}".format(i))
                print("z_den with supervised params {}".format(z_den.shape))

                temp_enc = concatenate([z_vel, z_den], axis=-1)
                print("temp_enc {}".format(temp_enc.shape))

                encoded = Lambda(lambda x: K.expand_dims(x, axis=1))(temp_enc)
                enc_input = concatenate([enc_input, encoded], axis=1) # (b, input_depth, z)

                # invalidate temp vars
                z_vel = None
                z_den = None
                sp_slice = None
                temp_enc = None
                enc_input_v = None
                enc_input_d = None
            print("INPUT SHAPE {} -> {}".format(i, enc_input))
            assert len(enc_input.shape) == 3, "enc_input shape does not match"

        print("enc_input {}".format(enc_input.shape))

        # directly extract z to apply supervised latent space loss afterwards
        print("z_vel_first {}".format(z_vel_first.shape))
        print("z_den_first {}".format(z_den_first.shape))

        rec_input = Lambda(lambda x: x[:, 0:self.w_num], name="rec_input_slice")(enc_input)
        print("rec_input {}".format(rec_input.shape))

        if self.ls_prediction_loss:
            rec_output_ls = None
        rec_output = None
        for i in range(self.recursive_prediction):
            print("\nPred_input_{}".format(i))
            print(rec_input.shape)
            x = pred([rec_input])
            print(x.shape)
            x = self.pred._fix_output_dimension(x)
            print(x.shape)

            # predicted delta 
            # add now to previous input
            pred_add_first_elem = Lambda(lambda x: x[:, -self.pred.out_w_num:None], name="rec_input_add_slice_{}".format(i))(rec_input)
            print("pred_add_first_elem: {}".format(pred_add_first_elem.shape))
            x = Add(name="Pred_Add_{}".format(i))([pred_add_first_elem, x]) # previous z + predicted delta z

            if self.ls_supervision:
                # set supervised part of rec_input to gt values
                real_x_sup = Lambda(lambda x: x[:, self.w_num + i], name="real_x_sup_{}".format(self.w_num + i))(sup_param_inputs)
                real_x_sup = Lambda(lambda x: K.expand_dims(x, axis=1))(real_x_sup)

                pred_x_v = Lambda(lambda x: x[:, :, 0:self.z_num_vel], name="pred_x_v_slice0_{}".format(i))(x)
                pred_x_d = Lambda(lambda x: x[:, :, self.z_num_vel:None], name="pred_x_d_slice0_{}".format(i))(x)

                assert pred_x_v.shape[2] == self.z_num_vel, "Dim does not fit v"
                assert pred_x_d.shape[2] == self.z_num_den, "Dim does not fit d"

                pred_x_v = Lambda(lambda x: x[:, :, 0:-self.sup_param_count], name="pred_x_v_slice1_{}".format(i))(pred_x_v)
                pred_x_d = Lambda(lambda x: x[:, :, 0:-self.sup_param_count], name="pred_x_d_slice1_{}".format(i))(pred_x_d)

                pred_x_v = concatenate([pred_x_v, real_x_sup], axis=2, name="Pred_Real_Supervised_v_Concat_{}".format(i))
                pred_x_d = concatenate([pred_x_d, real_x_sup], axis=2, name="Pred_Real_Supervised_d_Concat_{}".format(i))

                x = concatenate([pred_x_v, pred_x_d], axis=-1, name="Pred_Real_Supervised_Concat_{}".format(i))

            print("REC_INPUT_{}".format(i))
            print(rec_input.shape)
            
            rec_input = Lambda(lambda x: x[:, self.pred.out_w_num:None], name="rec_input_slice_{}".format(i))(rec_input)
            print(rec_input.shape)

            rec_input = concatenate([rec_input, x], axis=1, name="Pred_Input_Concat_{}".format(i))
            print(rec_input.shape)
            if self.decode_predictions:
                if self.ls_prediction_loss:
                    x_ls = x
                x_v = dec_v(pred_x_v)
                x_d = dec_d(pred_x_d)
                print("x_v decode {}".format(x_v.shape))
                print("x_d decode {}".format(x_d.shape))
                x = Concatenate(axis=-1, name="Concat_decoded_{}".format(i))(
                    [
                        x_v,
                        x_d
                    ])
            print("x decode {}".format(x.shape))

            if rec_output == None:
                rec_output = x
            else:
                rec_output = concatenate([rec_output, x], axis=1, name="Pred_Output_Concat_{}".format(i))

            if self.ls_prediction_loss:
                if rec_output_ls == None:
                    rec_output_ls = x_ls
                else:
                    rec_output_ls = concatenate([rec_output_ls, x_ls], axis=1, name="Pred_Output_LS_Concat_{}".format(i))

        print("rec_output {}".format(rec_output.shape))
        if self.decode_predictions:
            rec_output = Reshape((self.recursive_prediction,)+self.input_shape[1:], name="Reshape_decode_predictions")(rec_output)
            print("rec_output {}".format(rec_output.shape))

        if self.decode_predictions:
            GT_output = Lambda(lambda x: x[:, -self.recursive_prediction:None], name="GT_output_slice".format(i))(inputs)
        else:
            GT_output = Lambda(lambda x: x[:, -self.recursive_prediction:None], name="GT_output_encoded_slice".format(i))(enc_input)
        print("GT_output {}".format(GT_output.shape))

        # first half of pred_output is actual prediction, last half is GT to compare against in loss
        pred_output = concatenate([rec_output, GT_output], axis=1, name="Prediction_Output")
        print("pred_output {}".format(pred_output.shape))

        # supervised LS loss
        p_pred_vel = p_pred_v(Reshape((self.z_num_vel,), name="Reshape_pPred_vel")(z_vel_first))
        p_pred_den = p_pred_d(Reshape((self.z_num_den,), name="Reshape_pPred_den")(z_den_first))

        # decoder loss 
        ae_output_v = dec_v(z_vel_first)
        print("ae_output_v {}".format(ae_output_v.shape))
        ae_output_d = dec_d(z_den_first)
        print("ae_output_d {}".format(ae_output_d.shape))

        ae_combined = concatenate([ae_output_v, ae_output_d], axis=-1, name="outout")

        output_list = [ae_combined, pred_output, p_pred_vel, p_pred_den]

        print("Setup Model")
        if len(self.gpus) > 1:
            with tf.device('/cpu:0'):
                self.model = Model(name="Combined_AE_LSTM", inputs=[inputs, sup_param_inputs], outputs=output_list)
        else:
            self.model = Model(name="Combined_AE_LSTM", inputs=[inputs, sup_param_inputs], outputs=output_list)

    #---------------------------------------------------------------------------------
    def _compile_model(self):
        # loss_weights = [1.0, 1.0, 1.0]
        # if self.ls_prediction_loss:
        #     loss_weights.append(1.0)
        if len(self.gpus) > 1:
            self.parallel_model = multi_gpu_model(self.model, gpus=self.gpus)
            self.parallel_model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        else:
            self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics) # loss_weights=loss_weights, 

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
            self._build_model()
            self._compile_model()
        # Model Summary
        self.model.summary()
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
            train_generator = batch_manager.generator_ae_sequence_clean(batch_size, validation_split, validation=False, decode_predictions=self.decode_predictions, ls_prediction_loss=self.ls_prediction_loss)

            # validation samples
            val_gen_nb_samples = batch_manager.steps_per_epoch(batch_size, validation_split, validation=True)
            assert val_gen_nb_samples > 0, ("Batch size is too large for current scene samples/timestep settings. Training by generator not possible. Please adjust the batch size in the 'settings.json' file.")
            print ("Number of validation batch samples per epoch: {}".format(val_gen_nb_samples))
            validation_generator = batch_manager.generator_ae_sequence_clean(batch_size, validation_split, validation=True, decode_predictions=self.decode_predictions, ls_prediction_loss=self.ls_prediction_loss)

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

        from keras.utils.vis_utils import plot_model
        plot_model(self.model, to_file=self.model_dir+"/model_plot.png", show_shapes=True, show_layer_names=True)
        plot_model(self.ae_v._encoder, to_file=self.model_dir+"/enc_v_plot.png", show_shapes=True, show_layer_names=True)
        plot_model(self.ae_v._decoder, to_file=self.model_dir+"/dec_v_plot.png", show_shapes=True, show_layer_names=True)
        plot_model(self.ae_d._encoder, to_file=self.model_dir+"/enc_d_plot.png", show_shapes=True, show_layer_names=True)
        plot_model(self.ae_d._decoder, to_file=self.model_dir+"/dec_d_plot.png", show_shapes=True, show_layer_names=True)
        plot_model(self._p_pred_v, to_file=self.model_dir+"/p_pred_v_plot.png", show_shapes=True, show_layer_names=True)
        plot_model(self._p_pred_d, to_file=self.model_dir+"/p_pred_d_plot.png", show_shapes=True, show_layer_names=True)
        plot_model(self.pred.model, to_file=self.model_dir+"/pred_plot.png", show_shapes=True, show_layer_names=True)

    #---------------------------------------------------------------------------------
    def load_model(self, path, load_ae=True, load_pred=True):
        print("Loading model from {}".format(path))

        self._create_submodels()
        if load_ae:
            self.ae_v.load_model(path + "/ae_v/")
            self.ae_d.load_model(path + "/ae_d/")
        if load_pred:
            self.pred.load_model(path)

        if self.model is None:
            self._build_model()

    #---------------------------------------------------------------------------------
    def save_model(self, path):
        print("Saving model to {}".format(path))
        # store weights
        self.model.save_weights(path + "/combined_clean_ae_lstm.h5")
        self.ae_v.save_model(path + "/ae_v/")
        self.ae_d.save_model(path + "/ae_d/")
        self._p_pred_d.save_weights(path + "/p_pred_d_w.h5")
        self._p_pred_v.save_weights(path + "/p_pred_v_w.h5")
        self.pred.save_model(path)

    #---------------------------------------------------------------------------------
    def predict(self, x, batch_size=32):
        return self.model.predict(x, batch_size=batch_size)

    #---------------------------------------------------------------------------------
    def ae_predict(self, x, batch_size=32):
        velo_dim = 3 if self.is_3d else 2
        v_in = x[...,0:velo_dim]
        d_in = x[...,velo_dim:velo_dim+1]
        v, vp = self.ae_v.predict(v_in, batch_size=batch_size)
        d, dp = self.ae_d.predict(d_in, batch_size=batch_size)
        return [np.concatenate([v,d], axis=-1), v, d]

#---------------------------------------------------------------------------------
from config import get_config
from utils import prepare_dirs_and_logger
from keras_data import BatchManager
import os
from utils import save_image
from LatentSpacePhysics.src.util.requirements import init_packages
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

    os.makedirs( os.path.join(config.model_dir, "ae_d"), exist_ok=True)
    os.makedirs( os.path.join(config.model_dir, "checkpoint", "ae_d"), exist_ok=True)
    os.makedirs( os.path.join(config.model_dir, "ae_v"), exist_ok=True)
    os.makedirs( os.path.join(config.model_dir, "checkpoint", "ae_v"), exist_ok=True)

    # create GIT file
    repo = git.Repo(search_parent_directories=False)
    open("{}/{}".format(config.model_dir, repo.head.object.hexsha), "w") 

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

    keras_batch_manager = BatchManager(config, input_frame_count, prediction_window)
    sup_param_count = keras_batch_manager.supervised_param_count

    in_out_dim = 3 if "density" in config.data_type else 2
    in_out_dim = in_out_dim + 1 if config.is_3d else in_out_dim
    input_shape = (input_frame_count,)
    input_shape += (keras_batch_manager.res_z,) if config.is_3d else ()
    input_shape += (keras_batch_manager.res_y, keras_batch_manager.res_x, in_out_dim)

    print("Input Shape: {}".format(input_shape))

    rec_pred = RecursivePredictionCleanSplit(config=config, input_shape=input_shape, decode_predictions=decode_predictions, skip_pred_steps=skip_pred_steps, init_state_network=init_state_network, in_out_states=in_out_states, pred_gradient_loss=pred_gradient_loss, ls_prediction_loss=ls_prediction_loss, ls_supervision=ls_supervision, sqrd_diff_loss=sqrd_diff_loss, ls_split=ls_split, supervised_parameters=sup_param_count) 

    # Train =====================================================================================================
    if config.is_train:
        if config.load_path:
            rec_pred.load_model(config.load_path, load_ae=True, load_pred=True)
            
            rec_pred.pred._compile_model()
            rec_pred.ae._compile_model()
            rec_pred._compile_model()

        test_data = keras_batch_manager.batch_with_name(min(batch_num,8), validation_split=validation_split, validation=True)
        test_data = np.array(next(test_data)[0])
        print("test_data shape: {}".format(test_data.shape))
        if keras_batch_manager.res_z > 1:
            save_img_to_disk_3d(test_data, 0, config.model_dir, keras_batch_manager, "x_fixed_gt")
            plot_callback = PlotAEFields(rec_pred.ae_predict, test_data, save_img_to_disk_3d, config.model_dir, keras_batch_manager)
        else:
            save_img_to_disk(test_data, 0, config.model_dir, keras_batch_manager, "x_fixed_gt")
            plot_callback = PlotAEFields(rec_pred.ae_predict, test_data, save_img_to_disk, config.model_dir, keras_batch_manager)

        hist = rec_pred.train(epochs, batch_manager=keras_batch_manager, batch_size=batch_num, validation_split=validation_split, callbacks=[plot_callback])
        rec_pred.save_model(config.model_dir)
        rec_pred.print_summary()

        import LatentSpacePhysics.src.util.plot as plot
        import json
        if hist is not None:
            with open(config.model_dir+"/combined_clean_hist.json", 'w') as f:
                json.dump(hist.history, f, indent=4)

        # plot the history
        if hist:
            lstm_history_plotter = plot.Plotter()
            lstm_history_plotter.plot_history(hist.history)
            lstm_history_plotter.save_figures(config.model_dir+"/", "Combined_Clean_History", filetype="svg")
            lstm_history_plotter.save_figures(config.model_dir+"/", "Combined_Clean_History", filetype="png")

    # Test =====================================================================================================
    else:
        # ===============================================================================================
        # Check AE encdec after load
        rec_pred = RecursivePrediction(config=config, input_shape=input_shape, decode_predictions=decode_predictions, skip_pred_steps=skip_pred_steps, init_state_network=init_state_network, in_out_states=in_out_states, pred_gradient_loss=pred_gradient_loss, ls_prediction_loss=ls_prediction_loss, ls_supervision=ls_supervision, sqrd_diff_loss=sqrd_diff_loss, ls_split=ls_split, supervised_parameters=sup_param_count) 

        rec_pred.load_model("path to model checkpoint")

        ae = rec_pred.ae

        test_data = keras_batch_manager.batch_with_name(1, validation_split=1, validation=True, randomized=False)
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
