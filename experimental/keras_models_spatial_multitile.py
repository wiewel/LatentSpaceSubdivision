import sys,inspect,os
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import time

import tensorflow as tf

from keras_models_general import *

from LatentSpacePhysics.src.nn.stages import *
from LatentSpacePhysics.src.nn.helpers import *
from LatentSpacePhysics.src.nn.losses import *
from LatentSpacePhysics.src.nn.callbacks import LossHistory
from LatentSpacePhysics.src.nn.arch.architecture import Network

from ops import *
from os import path

import keras
from keras.optimizers import Adam
from keras import objectives
from keras.layers import *
from keras.layers.pooling import AveragePooling2D, MaxPooling2D
from keras.models import Model, save_model, load_model
from keras.callbacks import Callback
from keras.utils import multi_gpu_model
from keras.utils import CustomObjectScope
import keras.backend as K


#=====================================================================================
class Autoencoder(Network):
    #---------------------------------------------------------------------------------
    def _init_vars(self, config, settings=None, **kwargs):
        self.init_func = "glorot_normal"

        # Submodel Vars
        self._encoder = None
        self._decoder = None
        self._p_pred = None

        self.adam_epsilon = None 
        try:
            self.adam_learning_rate = config.lr
        except AttributeError:
            self.adam_learning_rate = 0.001
        try:
            self.adam_lr_decay = config.lr_decay
        except AttributeError:
            self.adam_lr_decay = 0.0005 
        try:
            self.vort_loss = config.vort_loss
        except AttributeError:
            self.vort_loss = False

        passive_input_loss = "density" in config.data_type or "levelset" in config.data_type
        self.input_shape = kwargs.get("input_shape", (64,64,64,3) if config.is_3d else (64,64,3))

        loss_list = [AE_Loss_Multitile(density=passive_input_loss, is_3d=config.is_3d, vort_loss=self.vort_loss), "mse"]
        self.loss_weights = [1.0,0.0] # disabled p_pred loss
        if config.tile_multitile_border > 0:
            loss_list.append("mae")
            self.loss_weights.append(1.0)
            loss_list.append("mae")
            self.loss_weights.append(1.0)
        self.set_loss(loss=loss_list)

        self.l1_reg = kwargs.get("l1_reg", 0.0)
        self.l2_reg = kwargs.get("l2_reg", 0.0)
        self.tensorflow_seed = kwargs.get("tensorflow_seed", 4)
        self.model = None
        tf.set_random_seed(self.tensorflow_seed)
        self.gpus = [ int(gpu.strip()) for gpu in config.gpu_id.split(",")]
        print("Using GPUs: {}".format(self.gpus))
        self.parallel_model = None

        self.stateful = kwargs.get("stateful", False)

        self.sup_param_count = kwargs.get("supervised_parameters", 1)

        self.name_prefix = kwargs.get("name_prefix", "")

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

        self.start_step = config.start_step
        self.max_step = config.max_step

        self.is_train = config.is_train

        self.dataset = config.dataset
        self.data_type = config.data_type

        self.b_num = config.batch_size
        self.z_num = config.z_num
        self.w_num = config.w_num

        try:
            self.keep_filter_size = config.keep_filter_size
            self.max_filters = config.max_filters
        except AttributeError:
            self.keep_filter_size = False
            self.max_filters = 512

        try:
            self.keep_filter_size_use_feature_add = config.keep_filter_size_use_feature_add
        except AttributeError:
            self.keep_filter_size_use_feature_add = False

        try:
            self.fully_conv = config.fully_conv
        except AttributeError:
            self.fully_conv = False

        self.repeat = config.repeat
        self.filters = config.filters
        self.num_conv = config.num_conv
        self.last_k = config.last_k
        self.skip_concat = config.skip_concat
        self.w1 = config.w1
        self.w2 = config.w2
        self.w_z = config.w_z
        self.tl = config.tl

        self.use_c = config.use_curl
        # if self.use_c:
        #     if self.is_3d:
        #         self.output_shape = get_conv_shape(self.x)[2:-1] + [3]
        #     else:
        #         self.output_shape = get_conv_shape(self.x)[2:-1] + [1]
        # else:
        #     self.output_shape = get_conv_shape(self.x)[2:]
        self.w_kl = config.w_kl
        self.sparsity = config.sparsity
        self.use_sparse = config.use_sparse


    #---------------------------------------------------------------------------------
    def set_loss(self, loss):
        self.loss = loss
        self.metrics = [] # ["mae"]
        # if not isinstance(self.loss, str):
        #     self.metrics = ["mse", "mae"]

    #---------------------------------------------------------------------------------
    def _init_optimizer(self, epochs=1):
        self.optimizer = Adam(lr=self.adam_learning_rate, epsilon=self.adam_epsilon, decay=self.adam_lr_decay)
        self.kernel_regularizer = regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg)
        return self.optimizer

    #---------------------------------------------------------------------------------
    def _create_submodels(self):
        # filters=64, z_num=32, num_conv=4, conv_k=3, last_k=3, repeat=0, output_shape=get_conv_shape(x)[1:]

        keep_filter_size = self.keep_filter_size
        max_filters = self.max_filters
        keep_filter_size_use_feature_add = self.keep_filter_size_use_feature_add
        filters = self.filters
        z_num = self.z_num
        num_conv_decoder = self.num_conv
        num_conv_encoder = num_conv_decoder - 1
        last_k = self.last_k
        repeat = self.repeat
        skip_concat = self.skip_concat
        is_3d = self.is_3d
        conv_k = 3

        act = None

        # ENCODER ################################################################################## 
        if not self._encoder:
            input_shape = self.input_shape
            if self.stateful:
                encoder_input = Input(batch_shape=(self.b_num,) + self.input_shape, name=self.name_prefix+"Encoder_Input") #shape=input_shape
            else:
                encoder_input = Input(shape=self.input_shape, name=self.name_prefix+"Encoder_Input") #shape=input_shape
            
            x = encoder_input

            x_shape = get_conv_shape(x)[1:]
            if repeat == 0:
                repeat_num = int(np.log2(np.max(x_shape[:-1]))) - 2 # q
            else:
                repeat_num = repeat
            assert(repeat_num > 0 and np.sum([i % np.power(2, repeat_num-1) for i in x_shape[:-1]]) == 0)
            
            ch = filters
            layer_num = 0
            x = conv_layer(x, filters=ch, kernel_size=conv_k, stride=1, activation=act, padding='same', kernel_initializer=self.init_func, kernel_regularizer=self.kernel_regularizer, is_3d=is_3d)

            x = LeakyReLU(0.3)(x)
            x0 = x
            layer_num += 1
            for idx in range(repeat_num):
                for _ in range(num_conv_encoder):
                    x = conv_layer(x, filters=filters, kernel_size=conv_k, stride=1, activation=act, padding='same', kernel_initializer=self.init_func, kernel_regularizer=self.kernel_regularizer, is_3d=is_3d)
                    x = LeakyReLU(0.3)(x)
                    layer_num += 1

                # skip connection
                if keep_filter_size and keep_filter_size_use_feature_add:
                    x = Add()([x, x0])
                else:
                    x = Concatenate(axis=-1)([x, x0])

                if not keep_filter_size:
                    ch += filters
                ch = min(ch, max_filters)

                if idx < repeat_num - 1:
                    x = conv_layer(x, filters=ch, kernel_size=conv_k, stride=2, activation=act, padding='same', kernel_initializer=self.init_func, kernel_regularizer=self.kernel_regularizer, is_3d=is_3d)
                    x = LeakyReLU(0.3)(x)
                    layer_num += 1
                    x0 = x

            if self.fully_conv:
                # # Version 1
                # last_encoder_shape = int_shape(x)
                # last_encoder_spatial_mult = np.prod(last_encoder_shape[1:-1])
                # print(last_encoder_shape)
                # print(last_encoder_spatial_mult)
                # last_encoder_feature_dim = int(z_num / last_encoder_spatial_mult)
                # assert last_encoder_feature_dim == int(z_num / last_encoder_spatial_mult + 0.5), "z_num must be divisible by spatial mutliplier!"

                # x = conv_layer(x, filters=last_encoder_feature_dim, kernel_size=1, stride=1, activation=act, padding='same', kernel_initializer=self.init_func, kernel_regularizer=self.kernel_regularizer, is_3d=is_3d)
                # x = LeakyReLU(0.3)(x)
                # last_encoder_shape = int_shape(x)

                # flat = Flatten()(x)
                # out = flat

                # Version 2
                x = conv_layer(x, filters=z_num, kernel_size=1, stride=1, activation=act, padding='same', kernel_initializer=self.init_func, kernel_regularizer=self.kernel_regularizer, is_3d=is_3d)
                last_encoder_shape = int_shape(x)

                x = GlobalMaxPooling3D()(x) if is_3d else GlobalMaxPooling2D()(x)
                #x = GlobalAveragePooling3D()(x) if is_3d else GlobalAveragePooling2D()(x)

                out = x
            else:
                flat = Flatten()(x)
                out = Dense(z_num, name=self.name_prefix+"z")(flat)

            # Flatten #
            print("Enc Out Shape: {}".format(x.shape[1:]))
            # final_z_num = 1
            # for cur_elem in x.shape[1:]:
            #     final_z_num *= cur_elem
            # final_z_num = int(final_z_num)
            # print("Final z: {}".format(final_z_num))

            self._encoder = Model(name=self.name_prefix+"Encoder", inputs=encoder_input, outputs=out)



        # DECODER ####################################################################################################
        if not self._decoder:
            decoder_input = Input(shape=(z_num,), name=self.name_prefix+"Decoder_Input")

            output_shape=get_conv_shape(encoder_input)[1:]

            if repeat == 0:
                repeat_num = int(np.log2(np.max(output_shape[:-1]))) - 2 #q
            else:
                repeat_num = repeat
            assert(repeat_num > 0 and np.sum([i % np.power(2, repeat_num-1) for i in output_shape[:-1]]) == 0)

            if self.fully_conv:
                x0_shape = last_encoder_shape[1:]
            else:
                x0_shape = [int(i/np.power(2, repeat_num-1)) for i in output_shape[:-1]] + [filters]

            num_output = int(np.prod(x0_shape))
            layer_num = 0

            if self.fully_conv:
                x = decoder_input
                x = RepeatVector(int(np.prod(x0_shape[:-1])))(x)
            else:
                x = Dense(num_output)(decoder_input)

            layer_num += 1
            x = Reshape(tuple(x0_shape))(x)

            if self.fully_conv:
                # deconv last encoder layer
                x = conv_layer(x, filters=filters, kernel_size=1, stride=1, activation=act, padding='same', kernel_initializer=self.init_func, kernel_regularizer=self.kernel_regularizer, is_3d=is_3d)
                x = LeakyReLU(0.3)(x)

            x0 = x

            for idx in range(repeat_num):
                for _ in range(num_conv_decoder):
                    x = conv_layer(x, filters=filters, kernel_size=conv_k, stride=1, activation=act, padding='same', kernel_initializer=self.init_func, kernel_regularizer=self.kernel_regularizer, is_3d=is_3d)
                    x = LeakyReLU(0.3)(x)
                    layer_num += 1

                if idx < repeat_num - 1:
                    if skip_concat:
                        x = upscale_layer(x, is_3d=is_3d)
                        x0 = upscale_layer(x0, is_3d=is_3d)
                        x = Concatenate(axis=-1)([x, x0])
                    else:
                        #x += x0
                        x = Add()([x,x0])
                        x = upscale_layer(x, is_3d=is_3d)
                        x0 = x
                elif not skip_concat:
                    #x += x0
                    x = Add()([x,x0])

            out = conv_layer(x, filters=output_shape[-1], kernel_size=last_k, stride=1, activation=act, padding='same', kernel_initializer=self.init_func, kernel_regularizer=self.kernel_regularizer, is_3d=is_3d)


            if "velocity" in self.config.data_type:
                if is_3d:
                    if self.use_c:
                        velo_out = Lambda(curl3)(out)
                        #_, velo_out = jacobian_layer(out, data_format='NHWC', is_3d=True)
                    else:
                        velo_out = out
                else:
                    if self.use_c:
                        velo_out = Lambda(curl, arguments={'data_format': 'NHWC'})(out)
                    else:
                        velo_out = out
                    # # comment in this part if curl layer is applied externally (outside of network)
                    # if self.use_c:
                    #     velo_out = Lambda(isolate_streamfunction, arguments={'data_format': 'NHWC'})(out)
                    # else:
                    #     velo_out = out

                if self.use_c and ("density" in self.config.data_type or "levelset" in self.config.data_type):
                    # cut of density part of "out" tensor and concatenate with the "velo_out" tensor
                    velo_out = Concatenate(axis=-1)([velo_out, Lambda(K.expand_dims, arguments={'axis': -1})(Lambda(lambda x: x[...,-1])(out))])
            else: 
                # only pressure or density given... direct output required
                velo_out = out

            print("Dec Out: {}".format(velo_out.shape))

            self._decoder = Model(name=self.name_prefix+"Decoder", inputs=decoder_input, outputs=velo_out)

        if not self._p_pred:
            pred_input = Input(shape=(self.z_num,))
            p_pred_out = Lambda(lambda x: x[:, -self.sup_param_count:], name=self.name_prefix+"p")(pred_input)
            self._p_pred = Model(name=self.name_prefix+"p_Pred", inputs=pred_input, outputs=p_pred_out)

    #---------------------------------------------------------------------------------
    def _build_model(self):
        self._create_submodels()

        # AUTOENCODER #########################################################################################
        outer_input_shape = list(self.input_shape)
        outer_input_shape[0] *= 2
        outer_input_shape[1] *= 2
        ae_input = Input(shape=outer_input_shape, dtype="float32", name=self.name_prefix+"Autoencoder_Input")

        if self.config.tiles_use_global:
            GT_fields = Lambda(lambda x: x[:, ..., 3:], name="Slice_gt_input_{}".format(0))(ae_input)
            GT_fields = AveragePooling2D(pool_size=(2, 2))(GT_fields)

        # 1. slice 
        cur_input = Lambda(lambda x: x[:, 0:self.config.res_y, 0:self.config.res_x, :3], name="Slice_ae_input_{}".format(0))(ae_input)
        if self.config.tiles_use_global:
            cur_input = concatenate([cur_input, GT_fields], axis=3)
        z0 = self._encoder(cur_input)
        output_0 = self._decoder(z0)
        # 2. slice 
        cur_input = Lambda(lambda x: x[:, 0:self.config.res_y, self.config.res_x:, :3], name="Slice_ae_input_{}".format(1))(ae_input)
        if self.config.tiles_use_global:
            cur_input = concatenate([cur_input, GT_fields], axis=3)
        z = self._encoder(cur_input)
        output_1 = self._decoder(z)
        # 3. slice 
        cur_input = Lambda(lambda x: x[:, self.config.res_y:, 0:self.config.res_x, :3], name="Slice_ae_input_{}".format(2))(ae_input)
        if self.config.tiles_use_global:
            cur_input = concatenate([cur_input, GT_fields], axis=3)
        z = self._encoder(cur_input)
        output_2 = self._decoder(z)
        # 4. slice 
        cur_input = Lambda(lambda x: x[:, self.config.res_y:, self.config.res_x:, :3], name="Slice_ae_input_{}".format(3))(ae_input)
        if self.config.tiles_use_global:
            cur_input = concatenate([cur_input, GT_fields], axis=3)
        z = self._encoder(cur_input)
        output_3 = self._decoder(z)

        final_out_0 = concatenate([output_0, output_1], axis=2)
        final_out_1 = concatenate([output_2, output_3], axis=2)
        final_out = concatenate([final_out_0, final_out_1], axis=1)

        # print("final_out: {}".format(final_out.shape))
        # if self.use_c:
        #     final_out_c = Lambda(curl, arguments={'data_format': 'NHWC'})(final_out)
        #     if self.use_c and ("density" in self.config.data_type or "levelset" in self.config.data_type):
        #         # cut of density part of "out" tensor and concatenate with the "velo_out" tensor
        #         final_out = Concatenate(axis=-1)([final_out_c, Lambda(K.expand_dims, arguments={'axis': -1})(Lambda(lambda x: x[...,-1])(final_out))])
        # print("use_c final_out: {}".format(final_out.shape))

        p_pred = self._p_pred(z0)
        print("p_pred: {}".format(p_pred.shape))

        output_list = [final_out, p_pred]

        if self.config.tile_multitile_border > 0:
            # Tile Loss
            border_region_start = (outer_input_shape[0] // 2) - 1 - self.config.tile_multitile_border
            border_region_end = (outer_input_shape[0] // 2) + self.config.tile_multitile_border
            tile_output0 = Lambda(lambda x: x[:, border_region_start:border_region_end, :, :], name="TileBorderLoss_{}".format(0))(final_out)
            border_region_start = (outer_input_shape[1] // 2) - 1 - self.config.tile_multitile_border
            border_region_end = (outer_input_shape[1] // 2) + self.config.tile_multitile_border
            tile_output1 = Lambda(lambda x: x[:, :, border_region_start:border_region_end, :], name="TileBorderLoss_{}".format(1))(final_out)
            output_list.append(tile_output0)
            output_list.append(tile_output1)

        if len(self.gpus) > 1:
            with tf.device('/cpu:0'):
                self.model = Model(name=self.name_prefix+"Autoencoder", inputs=ae_input, outputs=output_list)
        else:
            self.model = Model(name=self.name_prefix+"Autoencoder", inputs=ae_input, outputs=output_list)

    #---------------------------------------------------------------------------------
    def _compile_model(self):
        if len(self.gpus) > 1:
            self.parallel_model = multi_gpu_model(self.model, gpus=self.gpus)
            self.parallel_model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics, loss_weights=self.loss_weights)
        else:
            self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics, loss_weights=self.loss_weights)

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
        stateful = False
        batch_size = kwargs.get("batch_size", 8)
        history = None # not supported in stateful currently

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
            train_generator = batch_manager.generator_ae(batch_size, validation_split, validation=False, multitile=True)

            # validation samples
            val_gen_nb_samples = batch_manager.steps_per_epoch(batch_size, validation_split, validation=True)
            assert val_gen_nb_samples > 0, ("Batch size is too large for current scene samples/timestep settings. Training by generator not possible. Please adjust the batch size in the 'settings.json' file.")
            print ("Number of validation batch samples per epoch: {}".format(val_gen_nb_samples))
            validation_generator = batch_manager.generator_ae(batch_size, validation_split, validation=True, multitile=True)

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
                    # reset_callback = StatefulResetCallback(model)
                    # for i in range(epochs):
                    #     model.fit_generator(
                    #         generator=train_generator,
                    #         steps_per_epoch=train_gen_nb_samples, # how many batches to draw per epoch
                    #         epochs = 1,
                    #         verbose=1,
                    #         callbacks=[reset_callback],
                    #         validation_data=validation_generator,
                    #         validation_steps=val_gen_nb_samples,
                    #         class_weight=None,
                    #         workers=1)
                    #     model.reset_states()
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
                        workers=1)
            trainingDuration = time.time() - trainStartTime
        except KeyboardInterrupt:
            print("Training duration (s): {}\nInterrupted by user!".format(trainingDuration))
        print("Training duration (s): {}".format(trainingDuration))
        
        return history

    #---------------------------------------------------------------------------------
    def encoder_output_shape(self, input_shape=None):
        if not None in self.input_shape:
            input_shape = self.input_shape
        assert input_shape is not None, ("You must provide an input shape for autoencoders with variable input sizes")
        dummy_input = np.expand_dims(np.zeros(input_shape), axis = 0)
        shape = self.encode(dummy_input, 1).shape[1:]
        return shape

    #---------------------------------------------------------------------------------
    def print_summary(self):
        print("Autoencoder")
        self.model.summary()
        print("Encoder")
        self._encoder.summary()
        print("Decoder")
        self._decoder.summary()
        with open(path.join(self.config.model_dir, "model_summary.txt"),'w') as msf:
            self.model.summary(print_fn=lambda x: msf.write(x + "\n"))
        from keras.utils.vis_utils import plot_model
        plot_model(self.model, to_file=self.model_dir+"/model_plot.png", show_shapes=True, show_layer_names=True)
        plot_model(self._encoder, to_file=self.model_dir+"/enc_plot.png", show_shapes=True, show_layer_names=True)
        plot_model(self._decoder, to_file=self.model_dir+"/dec_plot.png", show_shapes=True, show_layer_names=True)

    #---------------------------------------------------------------------------------
    def load_model(self, path):
        print("Loading model from {}".format(path))

        self._create_submodels()

        with CustomObjectScope({'int_shape': int_shape, 'tf': tf}):
            # Load Encoder
            if os.path.exists(path + "/encoder_w.h5"):
                self._encoder.load_weights(path + "/encoder_w.h5", by_name=False)
            elif os.path.exists(path + "/encoder.h5"):
                temp_model = load_model(path + "/encoder.h5")
                self._encoder.set_weights(temp_model.get_weights())
            else: 
                print("WARNING: could not load weights for 'encoder'!")

            # Load Decoder
            if os.path.exists(path + "/decoder_w.h5"):
                self._decoder.load_weights(path + "/decoder_w.h5", by_name=False)
            elif os.path.exists(path + "/decoder.h5"):
                temp_model = load_model(path + "/decoder.h5")
                self._decoder.set_weights(temp_model.get_weights())
            else: 
                print("WARNING: could not load weights for 'decoder'!")

            # Load Pred
            if os.path.exists(path + "/p_pred_w.h5"):
                self._p_pred.load_weights(path + "/p_pred_w.h5", by_name=False)
            elif os.path.exists(path + "/p_pred.h5"):
                temp_model = load_model(path + "/p_pred.h5")
                self._p_pred.set_weights(temp_model.get_weights())
            else: 
                print("WARNING: could not load weights for 'p_pred'!")

        if self.model is None:
            self._build_model()


    #---------------------------------------------------------------------------------
    def save_model(self, path):
        print("Saving model to {}".format(path))
        # serialize model to JSON
        model_to_json(self._encoder, path + "/_encoder.json")
        model_to_json(self._decoder, path + "/_decoder.json")

        # store weights
        self.model.save_weights(path + "/autoencoder_w.h5")
        save_model(self._encoder, path + "/encoder.h5")
        save_model(self._decoder, path + "/decoder.h5")
        self._encoder.save_weights(path + "/encoder_w.h5")
        self._decoder.save_weights(path + "/decoder_w.h5")
        self._p_pred.save_weights(path + "/p_pred_w.h5")

    #---------------------------------------------------------------------------------
    def encode(self, x, batch_size=32):
        z = self._encoder.predict(x, batch_size=batch_size)
        return z

    #---------------------------------------------------------------------------------
    def decode(self, z, batch_size=32):
        y = self._decoder.predict(x=z, batch_size=batch_size)
        # # comment in if curl should be executed externally of network and is not part of the network itself
        # if self.use_c:
        #     y_c = curl_np(y)
        #     if ("density" in self.config.data_type or "levelset" in self.config.data_type):
        #         y = np.concatenate([y_c, np.expand_dims(y[...,-1], axis=-1)], axis=-1)
        #     else:
        #         y = y_c
        return y

    #---------------------------------------------------------------------------------
    def predict(self, x, batch_size=32):
        """ predict from three dimensional numpy array"""
        res = self.model.predict(x, batch_size=batch_size)
        return res

        # if self.use_c:
        #     print("res: {}".format(res[0].shape))
        #     res_c = curl_np(res[0])
        #     print("res_c: {}".format(res_c.shape))
        #     if ("density" in self.config.data_type or "levelset" in self.config.data_type):
        #         res[0] = np.concatenate([res_c, np.expand_dims(res[0][...,-1], axis=-1)], axis=-1)
        #     else:
        #         res[0] = res_c
        #     print("res[0]: {}".format(res[0].shape))
        # return res


from config import get_config
from utils import prepare_dirs_and_logger
from keras_data import BatchManager, copy_dataset_info
import os
from utils import save_image
from LatentSpacePhysics.src.util.requirements import init_packages
init_packages()

#---------------------------------------------------------------------------------
def get_vort(x, batch_manager, is_vel=False):
    is_3d = batch_manager.is_3d
    if is_vel:
        if is_3d:
            _, x = jacobian_np3(x[:,:,:,:,:3])
        else:
            x = vort_np(x[:,:,:,:2])
    else:        
        # x range [-1, 1], NHWC
        if is_3d:
            _, x = jacobian_np3(x[:,:,:,:,:3]) # streamfunction to velocity
            _, x = jacobian_np3(x[:,:,:,:,:3]) # velocity to vorticity
        else:
            x = vort_np(curl_np(x))
        x = batch_manager.to_vel(x)
    return x

#---------------------------------------------------------------------------------
def save_img_to_disk(img, idx, root_path, batch_manager, name="", img_vort=None):
    single_img = False
    img_den = None
    if(img.shape[-1] > 2):
        img_den = img[...,2]
        img_den = np.expand_dims(img_den, axis=-1)
        img_den = denorm_img_numpy(img_den)
        img_den = np.concatenate((img_den,img_den,img_den), axis=3)
        img = img[...,0:2]
    img = denorm_img_numpy(img)
    if img_vort is None and img.shape[-1] >= 2:
        img_vort = get_vort(img / 127.5 - 1, batch_manager=batch_manager, is_vel=True)
        img_vort = denorm_img_numpy(img_vort)
        img_vort = np.concatenate((img_vort,img_vort,img_vort), axis=3)
    if img_vort is not None:
        img = np.concatenate((img,img_vort), axis=0)
    if img_den is not None:
        if img.shape[-1] == 1:
            img = img_den
            single_img = True
        else:
            img = np.concatenate((img, img_den), axis=0)
    path = os.path.join(root_path, '{}_{}.png'.format(name, idx))
    save_image(img, path, single=single_img)
    print("[*] Samples saved: {}".format(path))

#---------------------------------------------------------------------------------
def save_img_to_disk_3d(img, idx, root_path, batch_manager, name="", img_vort=None):
    img_den = None
    if(img.shape[-1] > 3): # if more channels than velocity components exist 
        img_den = img[...,-1]
        img_den = np.expand_dims(img_den, axis=-1)
        # select best view here
        img_den = denorm_img3_numpy(img_den)['xym']
        img_den = np.concatenate((img_den,img_den,img_den), axis=3)
        img = img[...,0:3] # remove density part

    if img_vort is None:
        img_vort = get_vort(img, batch_manager=batch_manager, is_vel=True)
        img_vort = denorm_img3_numpy(img_vort)
        # select best view here
        img_vort = img_vort['xym']

    # select best view here
    img = denorm_img3_numpy(img)
    img = img['xym']
    img = np.concatenate((img,img_vort), axis=0)

    if img_den is not None:
        img = np.concatenate((img, img_den), axis=0)

    path = os.path.join(root_path, '{}_{}.png'.format(idx, name))
    save_image(img, path)
    print("[*] Samples saved: {}".format(path))

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

    # copy dataset info to model dir
    copy_dataset_info(config)

    # Transfer data to local vars
    batch_num = config.batch_size
    validation_split = 0.1
    epochs = config.epochs
    sqrd_diff_loss = config.sqrd_diff_loss
    ls_split = config.ls_split
    test_data_types = config.data_type.copy()
    if "inflow" in test_data_types: test_data_types.remove("inflow")

    # Multitile model -> scale by 2
    config.tile_scale = 2

    keras_batch_manager = BatchManager(config, 1, 1)
    sup_param_count = keras_batch_manager.supervised_param_count

    in_out_dim = 3 if "density" in config.data_type or "levelset" in config.data_type else 2
    in_out_dim = in_out_dim + 1 if config.is_3d else in_out_dim

    try:
        tiles_use_global = config.tiles_use_global
    except AttributeError:
        tiles_use_global = False
    if keras_batch_manager.use_tiles and tiles_use_global:
        # extra channels for global information
        in_out_dim += 2

    input_shape = (config.res_z,) if config.is_3d else ()
    input_shape += (config.res_y, config.res_x, in_out_dim)

    print("Input Shape: {}".format(input_shape))

    ae = Autoencoder(config=config, input_shape=input_shape, supervised_parameters=sup_param_count)

    # Train =====================================================================================================
    if config.is_train:
        if config.load_path:
            ae.load_model(config.load_path)

        test_data = keras_batch_manager.batch_with_name(min(batch_num,8), validation_split=validation_split, validation=True, use_tiles=keras_batch_manager.tile_generator is not None)
        test_data = np.array(next(test_data)[0])
        print("test_data shape: {}".format(test_data.shape))
        if keras_batch_manager.is_3d:
            save_img_to_disk_3d(test_data, 0, config.model_dir, keras_batch_manager, "x_fixed_gt")
            plot_callback = PlotAEFields(ae.predict, test_data, save_img_to_disk_3d, config.model_dir, keras_batch_manager)
        else:
            save_img_to_disk(test_data, 0, config.model_dir, keras_batch_manager, "x_fixed_gt")
            plot_callback = PlotAEFields(ae.predict, test_data, save_img_to_disk, config.model_dir, keras_batch_manager)

        hist = ae.train(epochs, batch_size=batch_num, batch_manager=keras_batch_manager, callbacks=[plot_callback], validation_split=validation_split, embedding_data=test_data)
        ae.save_model(config.model_dir)
        ae.print_summary()

        import LatentSpacePhysics.src.util.plot as plot
        import json
        if hist is not None:
            with open(config.model_dir+"/spatial_hist.json", 'w') as f:
                json.dump(hist.history, f, indent=4)

        # plot the history
        if hist:
            history_plotter = plot.Plotter()
            history_plotter.plot_history(hist.history)
            history_plotter.save_figures(config.model_dir+"/", "Spatial_History", filetype="svg")
            history_plotter.save_figures(config.model_dir+"/", "Spatial_History", filetype="png")

    # Test =====================================================================================================
    else:
        ae = Autoencoder(config=config, input_shape=input_shape, supervised_parameters=sup_param_count)

        if config.load_path:
            ae.load_model(config.load_path)

        test_data = keras_batch_manager.batch_with_name(1, validation_split=1, validation=True, randomized=True, use_tiles=False)

        test_data = np.array(next(test_data)[0])
        print("test_data shape: {}".format(test_data.shape))

        full_res_image = np.zeros_like(test_data, dtype=np.float32)
        print("full_res_image shape: {}".format(full_res_image.shape))

        plot_callback = PlotAEFields(ae.predict, None, save_img_to_disk, "./test/", keras_batch_manager)

        from skimage import measure

        if keras_batch_manager.tile_generator is not None:
            while keras_batch_manager.tile_generator.getNextTile():
                keras_batch_manager.tile_generator.print()
                tile = keras_batch_manager.tile_generator.cut_tile_2d(test_data)
                print("tile shape: {}".format(tile.shape))
                if tiles_use_global:
                    x_mult = int(keras_batch_manager.tile_generator.data_dim[0] / keras_batch_manager.tile_generator.tile_size[0])
                    y_mult = int(keras_batch_manager.tile_generator.data_dim[1] / keras_batch_manager.tile_generator.tile_size[1])
                    z_mult = int(keras_batch_manager.tile_generator.data_dim[2] / keras_batch_manager.tile_generator.tile_size[2])
                    tile_flag_shape = list(test_data.shape)
                    tile_flag_shape[-1] = 1
                    tile_flag = np.zeros(tile_flag_shape)
                    tile_flag[keras_batch_manager.tile_generator.y_start:keras_batch_manager.tile_generator.y_end, keras_batch_manager.tile_generator.x_start:keras_batch_manager.tile_generator.x_end, :] = 1
                    # 2:3 -> capture only density part
                    x_downscale = measure.block_reduce(test_data[...,2:3], (1, y_mult, x_mult, 1), np.mean)
                    tile_flag_downscale = measure.block_reduce(tile_flag, (1, y_mult, x_mult, 1), np.mean)
                    tile = np.append(tile, x_downscale, axis=-1)
                    tile = np.append(tile, tile_flag_downscale, axis=-1)
                plot_callback._x = tile
                plot_callback.on_epoch_end(0,0)
                full_res_image[...,keras_batch_manager.tile_generator.y_start:keras_batch_manager.tile_generator.y_end,
                keras_batch_manager.tile_generator.x_start:keras_batch_manager.tile_generator.x_end, :] = plot_callback._y
            # if x[0].ndim == 4:
            #     x_tile = x[self.tile_generator.z_start:self.tile_generator.z_end, self.tile_generator.y_start:self.tile_generator.y_end, self.tile_generator.x_start:self.tile_generator.x_end, :]
            # else:
            #     x_tile = x[self.tile_generator.y_start:self.tile_generator.y_end, self.tile_generator.x_start:self.tile_generator.x_end, :]
            # x = x_tile
        print("full_res_image shape: {}".format(full_res_image.shape))
        save_img_to_disk(full_res_image, 0, "./test/", keras_batch_manager, "EncDec")
        save_img_to_disk(test_data, 0, "./test/", keras_batch_manager, "GT")
