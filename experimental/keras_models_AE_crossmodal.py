import sys,inspect,os
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from keras_models_combined import *

#=====================================================================================
class CrossModalAE(Network):
    #---------------------------------------------------------------------------------
    def _init_vars(self, config, settings=None, **kwargs):
        self.config = config
        self.kwargs = kwargs

        # Submodel Vars
        self.ae_v = None
        self.ae_d = None
        self._p_pred_d = None
        self._p_pred_v = None

        self.adam_epsilon = None
        try:
            self.adam_learning_rate = config.lr
        except AttributeError:
            self.adam_learning_rate = 0.001
        try:
            self.adam_lr_decay = config.lr_decay
        except AttributeError:
            self.adam_lr_decay = 0.0005

        self.sqrd_diff_loss = kwargs.get("sqrd_diff_loss", False)
        self.ls_split = kwargs.get("ls_split", 0.0)
        self.sup_param_count = kwargs.get("supervised_parameters", 1)
        self.z_num = config.z_num
        self.is_3d = config.is_3d
        self.model_dir = config.model_dir

        self.l1_reg = kwargs.get("l1_reg", 0.0)
        self.l2_reg = kwargs.get("l2_reg", 0.0)
        self.tensorflow_seed = kwargs.get("tensorflow_seed", 4)
        self.model = None
        tf.set_random_seed(self.tensorflow_seed)
        self.gpus = [ int(gpu.strip()) for gpu in config.gpu_id.split(",")]
        print("Using GPUs: {}".format(self.gpus))
        self.parallel_model = None

        use_density = "density" in self.config.data_type

        # TODO: potentially add losses here for crossmodal latent space
        loss_list = [
            AE_Loss(sqrd_diff_loss=self.sqrd_diff_loss, density=use_density, is_3d=self.is_3d),
            "mse",
            "mse",
            AE_Loss(sqrd_diff_loss=self.sqrd_diff_loss, density=use_density, is_3d=self.is_3d)
            # dec_v(c_d) - v
            # dec_d(c_v) - d
        ]

        self.set_loss(loss=loss_list)

    #---------------------------------------------------------------------------------
    def set_loss(self, loss):
        self.loss = loss
        self.metrics = None

    #---------------------------------------------------------------------------------
    def _init_optimizer(self, epochs=1):
        self.optimizer = Adam(lr=self.adam_learning_rate, epsilon=self.adam_epsilon, decay=self.adam_lr_decay)
        self.kernel_regularizer = regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg)
        return self.optimizer

    #---------------------------------------------------------------------------------
    def _create_submodels(self):
        if not self.ae_v:
            kwargs_copy = self.kwargs
            print("Velo AE")
            print(kwargs_copy["input_shape"])
            in_v = list(kwargs_copy["input_shape"])
            in_v[-1] = in_v[-1] - 1
            kwargs_copy["input_shape"] = tuple(in_v)
            print(kwargs_copy["input_shape"])
            kwargs_copy["name_prefix"] = "AE_v"
            config_copy = self.config
            config_copy.data_type = "velocity"
            self.ae_v = Autoencoder(config=config_copy, **kwargs_copy)
            self.ae_v._build_model()
        if not self.ae_d:
            kwargs_copy = self.kwargs
            print("Den AE")
            print(kwargs_copy["input_shape"])
            in_d = list(kwargs_copy["input_shape"])
            in_d[-1] = 1
            kwargs_copy["input_shape"] = tuple(in_d)
            print(kwargs_copy["input_shape"])
            kwargs_copy["name_prefix"] = "AE_d"
            config_copy = self.config
            config_copy.data_type = "density"
            self.ae_d = Autoencoder(config=config_copy, **kwargs_copy)
            self.ae_d._build_model()
        if not self._p_pred_v:
            pred_input = Input(shape=(self.z_num,))
            p_pred_out = Lambda(lambda x: x[:, -self.sup_param_count:], name="p_vel")(pred_input)
            self._p_pred_v = Model(name="p_Pred_v", inputs=pred_input, outputs=p_pred_out)
        if not self._p_pred_d:
            pred_input = Input(shape=(self.z_num,))
            p_pred_out = Lambda(lambda x: x[:, -self.sup_param_count:], name="p_den")(pred_input)
            self._p_pred_d = Model(name="p_Pred_d", inputs=pred_input, outputs=p_pred_out)

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

        input_shape = list(self.ae_v.input_shape)
        input_shape[-1] = input_shape[-1] + 1
        print("Input Shape: {} -> {}".format(self.ae_v.input_shape, input_shape))
        inputs_full = Input(shape=input_shape, dtype="float32", name="CrossModal_Input")

        inputs_vel = Lambda(lambda x: x[...,0:velo_dim], name="Vel_Input")(inputs_full)
        inputs_den = Lambda(lambda x: x[...,velo_dim:velo_dim+1], name="Den_Input")(inputs_full)
        print("inputs_vel {}".format(inputs_vel.shape))
        print("inputs_den {}".format(inputs_den.shape))

        z_vel = enc_v(inputs_vel)
        z_vel = Lambda(lambda x: x, name="z_vel")(z_vel)

        z_den = enc_d(inputs_den)
        z_den = Lambda(lambda x: x, name="z_den")(z_den)

        print("z_vel {}".format(z_vel.shape))
        print("z_den {}".format(z_den.shape))

        # supervised LS loss
        p_pred_vel = p_pred_v(Reshape((self.z_num,), name="Reshape_pPred_vel")(z_vel))
        p_pred_den = p_pred_d(Reshape((self.z_num,), name="Reshape_pPred_den")(z_den))

        # Classic loss d - d_hat; v - v_hat
        decoded_v = dec_v(z_vel)
        print("decoded_v {}".format(decoded_v.shape))
        decoded_d = dec_d(z_den)
        print("decoded_d {}".format(decoded_d.shape))
        decoded_full = Concatenate(axis=-1, name="EncDec")(
            [
                decoded_v,
                decoded_d
            ])

        # Cross Modal loss dec_d(c_v) - d; dec_v(c_d) - v
        decoded_v_cm = dec_v(z_den)
        print("decoded_v_cm {}".format(decoded_v_cm.shape))
        decoded_d_cm = dec_d(z_vel)
        print("decoded_d_cm {}".format(decoded_d_cm.shape))
        decoded_full_cm = Concatenate(axis=-1, name="CrossModal_EncDec")(
            [
                decoded_v_cm,
                decoded_d_cm
            ])

        output_list = [decoded_full, p_pred_vel, p_pred_den, decoded_full_cm]

        print("Setup Model")
        if len(self.gpus) > 1:
            with tf.device('/cpu:0'):
                self.model = Model(name="CrossModal_AE", inputs=[inputs_full], outputs=output_list)
        else:
            self.model = Model(name="CrossModal_AE", inputs=[inputs_full], outputs=output_list)


    #---------------------------------------------------------------------------------
    def _compile_model(self):
        if len(self.gpus) > 1:
            self.parallel_model = multi_gpu_model(self.model, gpus=self.gpus)
            self.parallel_model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        else:
            self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

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
        stateful = False
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
            train_generator = batch_manager.generator_ae_crossmodal(batch_size, validation_split, validation=False)

            # validation samples
            val_gen_nb_samples = batch_manager.steps_per_epoch(batch_size, validation_split, validation=True)
            assert val_gen_nb_samples > 0, ("Batch size is too large for current scene samples/timestep settings. Training by generator not possible. Please adjust the batch size in the 'settings.json' file.")
            print ("Number of validation batch samples per epoch: {}".format(val_gen_nb_samples))
            validation_generator = batch_manager.generator_ae_crossmodal(batch_size, validation_split, validation=True)

        try:
            trainingDuration = 0.0
            trainStartTime = time.time()
            if stateful:
                if (batch_manager is None):
                    assert X is not None and Y is not None, ("X or Y is None!")
                    for i in range(epochs):
                        model.fit(
                            X,
                            Y,
                            epochs=1,
                            batch_size=batch_size,
                            shuffle=False,
                            callbacks=callbacks)
                        model.reset_states()
                else:
                    assert False, ("Not implemented yet")
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

    #---------------------------------------------------------------------------------
    def load_model(self, path):
        print("Loading model from {}".format(path))

        self._create_submodels()
        self.ae_v.load_model(path + "/ae_v/")
        self.ae_d.load_model(path + "/ae_d/")

        if self.model is None:
            self._build_model()

    #---------------------------------------------------------------------------------
    def save_model(self, path):
        print("Saving model to {}".format(path))
        # store weights
        self.model.save_weights(path + "/crossmodal_ae_w.h5")
        self.ae_v.save_model(path + "/ae_v/")
        self.ae_d.save_model(path + "/ae_d/")
        self._p_pred_d.save_weights(path + "/p_pred_d_w.h5")
        self._p_pred_v.save_weights(path + "/p_pred_v_w.h5")

    #---------------------------------------------------------------------------------
    def predict(self, x, batch_size=32):
        return self.model.predict(x, batch_size=batch_size)

    #---------------------------------------------------------------------------------
    def ae_predict(self, x, batch_size=32):
        ae_pred = self.model.predict(x, batch_size=32)
        return ae_pred

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
    sqrd_diff_loss = config.sqrd_diff_loss
    ls_split = config.ls_split

    keras_batch_manager = BatchManager(config, 1, 1)
    sup_param_count = keras_batch_manager.supervised_param_count

    in_out_dim = 3 if "density" in config.data_type else 2
    in_out_dim = in_out_dim + 1 if config.is_3d else in_out_dim

    input_shape = (keras_batch_manager.res_z,) if config.is_3d else ()
    input_shape += (keras_batch_manager.res_y, keras_batch_manager.res_x, in_out_dim)

    print("Input Shape: {}".format(input_shape))

    ae = CrossModalAE(config=config, input_shape=input_shape, sqrd_diff_loss=sqrd_diff_loss, supervised_parameters=sup_param_count)

    # Train =====================================================================================================
    if config.is_train:
        if config.load_path:
            ae.load_model(config.load_path)

        test_data = keras_batch_manager.batch_with_name(min(batch_num,8), validation_split=validation_split, validation=True)
        test_data = np.array(next(test_data)[0])
        print("test_data shape: {}".format(test_data.shape))
        if keras_batch_manager.res_z > 1:
            save_img_to_disk_3d(test_data, 0, config.model_dir, keras_batch_manager, "x_fixed_gt")
            plot_callback = PlotAEFields(ae.ae_predict, test_data, save_img_to_disk_3d, config.model_dir, keras_batch_manager)
        else:
            save_img_to_disk(test_data, 0, config.model_dir, keras_batch_manager, "x_fixed_gt")
            plot_callback = PlotAEFields(ae.ae_predict, test_data, save_img_to_disk, config.model_dir, keras_batch_manager)

        hist = ae.train(epochs, batch_manager=keras_batch_manager, batch_size=batch_num, validation_split=validation_split, callbacks=[plot_callback])
        ae.save_model(config.model_dir)
        ae.print_summary()

        import LatentSpacePhysics.src.util.plot as plot
        import json
        if hist is not None:
            with open(config.model_dir+"/CrossModal_AE_hist.json", 'w') as f:
                json.dump(hist.history, f, indent=4)

        # plot the history
        if hist:
            lstm_history_plotter = plot.Plotter()
            lstm_history_plotter.plot_history(hist.history)
            lstm_history_plotter.save_figures(config.model_dir+"/", "CrossModal_AE_History", filetype="svg")
            lstm_history_plotter.save_figures(config.model_dir+"/", "CrossModal_AE_History", filetype="png")

    # Test =====================================================================================================
    else:
        ae = CrossModalAE(config=config, input_shape=(keras_batch_manager.res_y,keras_batch_manager.res_x,in_out_dim))
        ae._build_model()
        ae.load_model(config.load_path)

        test_data = keras_batch_manager.batch_with_name(batch_num, validation_split=validation_split, validation=True, randomized=False)
        encoded_data = []
        for x, path, _ in test_data:
            enc = ae.encode(x=np.asarray(x), batch_size=batch_num)
            encoded_data.append(enc)

        encoded_data = np.asarray(encoded_data).reshape((keras_batch_manager.num_scenes, keras_batch_manager.num_frames, config.z_num))
        print(encoded_data.shape)

        if config.is_3d:
            dim_sup = 2
        else:
            dim_sup = 1

        from LatentSpacePhysics.src.dataset import datasets
        import datetime
        encoded_scene_list = datasets.SceneList(encoded_data, scene_size=keras_batch_manager.num_frames, version=dim_sup, date=str(datetime.datetime.now()), name="encoded_deep_fluid_{}_{}".format(config.z_num, config.dataset), seed=0, test_split=0.05)

        dataset_path = config.load_path
        print("Enc Scene List Path: {}".format(dataset_path))

        encoded_scene_list.serialize(dataset_path=dataset_path)
        histogram_path = dataset_path if dataset_path else "."
        encoded_scene_list.save_histogram(histogram_path + "/enc_scene_list_{}_normalized_{}_{}.png".format(encoded_scene_list.name, config.z_num, config.dataset))
