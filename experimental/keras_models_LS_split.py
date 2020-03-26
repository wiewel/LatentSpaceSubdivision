import sys,inspect,os
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from keras_models_combined import *

#---------------------------------------------------------------------------------------------------------------------------
class Zero_Loss(object):
    #---------------------------------------------------------------------------------
    def __init__(self):
        self.__name__ = "Zero_Loss"
    #---------------------------------------------------------------------------------
    def __call__(self, y_true, y_pred):
        return y_true-y_true

#=====================================================================================
class LatentSpaceSplit(Network):
    #---------------------------------------------------------------------------------
    def _init_vars(self, config, settings=None, **kwargs):
        self.config = config
        self.kwargs = kwargs

        # Submodel Vars
        self.ae = None

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

        if self.sup_param_count > 0:
            loss_list = [AE_Loss(sqrd_diff_loss=self.sqrd_diff_loss, density=use_density, is_3d=self.is_3d), "mse"]
        else:
            loss_list = [AE_Loss(sqrd_diff_loss=self.sqrd_diff_loss, density=use_density, is_3d=self.is_3d), Zero_Loss()]

        if self.ls_split > 0.0:
            self.ls_split_idx = int(self.z_num * self.ls_split)
            assert self.ls_split_idx > 0, "ls_split_idx must be larger than 0!"

            loss_list.append(Split_Loss(self.ls_split_idx, self.z_num - self.sup_param_count)) # one is skipped for supervised parameter
            loss_list.append(Split_Loss(0, self.ls_split_idx))
            print("Splitting LS at  0 -> {} and {} -> {}".format(self.ls_split_idx, self.ls_split_idx, self.z_num - self.sup_param_count))

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
        if not self.ae:
            self.ae = Autoencoder(config=self.config, **self.kwargs)
            self.ae._build_model()

    #---------------------------------------------------------------------------------
    def _build_model(self):
        print("Building Model")

        velo_dim = 3 if self.is_3d else 2

        self._create_submodels()

        enc = self.ae._encoder
        dec = self.ae._decoder
        p_pred = self.ae._p_pred 

        inputs_full = Input(shape=self.ae.input_shape, dtype="float32", name="LatentSpaceSplit_Input")
        inputs_vel = Lambda(lambda x: K.concatenate([x[...,0:velo_dim], K.zeros_like(x)[...,velo_dim:velo_dim+1]], axis=-1), name="ls_split_1")(inputs_full)
        inputs_den = Lambda(lambda x: K.concatenate([K.zeros_like(x)[...,0:velo_dim], x[...,velo_dim:velo_dim+1]], axis=-1), name="ls_split_2")(inputs_full)
        print("inputs_vel {}".format(inputs_vel.shape))
        print("inputs_den {}".format(inputs_den.shape))

        z_full = enc(inputs_full)
        z_vel = enc(inputs_vel)
        z_vel = Lambda(lambda x: x, name="z_vel")(z_vel)
        z_den = enc(inputs_den)
        z_den = Lambda(lambda x: x, name="z_den")(z_den)
        print("z_full {}".format(z_full.shape))
        print("z_vel {}".format(z_vel.shape))
        print("z_den {}".format(z_den.shape))

        # supervised LS loss
        p_pred_output = p_pred(Reshape((self.z_num,), name="Reshape_pPred")(z_full))

        ae_output = dec(z_full)
        print("ae_output {}".format(ae_output.shape))

        output_list = [ae_output, p_pred_output, z_vel, z_den]

        print("Setup Model")
        if len(self.gpus) > 1:
            with tf.device('/cpu:0'):
                self.model = Model(name="Split_LS_AE", inputs=[inputs_full], outputs=output_list)
        else:
            self.model = Model(name="Split_LS_AE", inputs=[inputs_full], outputs=output_list)

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
            train_generator = batch_manager.generator_ae_split(batch_size, validation_split, validation=False)

            # validation samples
            val_gen_nb_samples = batch_manager.steps_per_epoch(batch_size, validation_split, validation=True)
            assert val_gen_nb_samples > 0, ("Batch size is too large for current scene samples/timestep settings. Training by generator not possible. Please adjust the batch size in the 'settings.json' file.")
            print ("Number of validation batch samples per epoch: {}".format(val_gen_nb_samples))
            validation_generator = batch_manager.generator_ae_split(batch_size, validation_split, validation=True)

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
        plot_model(self.ae._encoder, to_file=self.model_dir+"/enc_plot.png", show_shapes=True, show_layer_names=True)
        plot_model(self.ae._decoder, to_file=self.model_dir+"/dec_plot.png", show_shapes=True, show_layer_names=True)
        plot_model(self.ae._p_pred, to_file=self.model_dir+"/p_pred_plot.png", show_shapes=True, show_layer_names=True)

    #---------------------------------------------------------------------------------
    def load_model(self, path):
        print("Loading model from {}".format(path))

        self._create_submodels()
        self.ae.load_model(path)

        if self.model is None:
            self._build_model()

    #---------------------------------------------------------------------------------
    def save_model(self, path):
        print("Saving model to {}".format(path))
        # store weights
        self.model.save_weights(path + "/split_loss_ae.h5")
        self.ae.save_model(path)

    #---------------------------------------------------------------------------------
    def predict(self, x, batch_size=32):
        return self.model.predict(x, batch_size=batch_size)

    #---------------------------------------------------------------------------------
    def ae_predict(self, x, batch_size=32):
        return self.ae.predict(x, batch_size=batch_size)

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
    sqrd_diff_loss = config.sqrd_diff_loss
    ls_split = config.ls_split
    no_sup_params = config.no_sup_params

    keras_batch_manager = BatchManager(config, 1, 1)
    sup_param_count = 0 if no_sup_params else keras_batch_manager.supervised_param_count 

    in_out_dim = 3 if "density" in config.data_type else 2
    in_out_dim = in_out_dim + 1 if config.is_3d else in_out_dim

    input_shape = (keras_batch_manager.res_z,) if config.is_3d else ()
    input_shape += (keras_batch_manager.res_y, keras_batch_manager.res_x, in_out_dim)

    print("Input Shape: {}".format(input_shape))

    ae = LatentSpaceSplit(config=config, input_shape=input_shape, sqrd_diff_loss=sqrd_diff_loss, ls_split=ls_split, supervised_parameters=sup_param_count)

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
            with open(config.model_dir+"/LS_split_hist.json", 'w') as f:
                json.dump(hist.history, f, indent=4)

        # plot the history
        if hist:
            lstm_history_plotter = plot.Plotter()
            lstm_history_plotter.plot_history(hist.history)
            lstm_history_plotter.save_figures(config.model_dir+"/", "LS_split_History", filetype="svg")
            lstm_history_plotter.save_figures(config.model_dir+"/", "LS_split_History", filetype="png")

    # Test =====================================================================================================
    else:
        ae = LatentSpaceSplit(config=config, input_shape=(keras_batch_manager.res_y,keras_batch_manager.res_x,in_out_dim))
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
