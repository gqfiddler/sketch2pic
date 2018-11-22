import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
import pickle
import tensorflow as tf
from tensorflow.keras import layers, regularizers, losses
from tensorflow.keras.models import Model, Sequential
import tensorflow.keras.backend as K

class sketch2pic():

    def __init__(self, name, lambda_=50, generator_activation='tanh'):

        self.name = name
        self.lambda_ = lambda_
        self.gen_activation = generator_activation
        self.previous_best = list(np.zeros(100))

        class History():
            def __init__(self, model_name):
                self.model_name = model_name
                self.checkpoints = [] # measured in epochs or fractions of epochs

                self.train_loss = []
                self.dev_loss = []
                self.train_acc_real = []
                self.train_acc_fake = []
                self.dev_acc_real = []
                self.dev_acc_fake = []

                self.dev_images = []
                self.dev_sketches = []
                self.dev_fakes = []
                self.dev_fakes_first_epoch = []

            def update(self, checkpoint, train_loss, dev_loss, \
                    train_acc_real, train_acc_fake, dev_acc_real, dev_acc_fake):
                self.checkpoints.append(checkpoint)
                self.train_loss.append(train_loss)
                self.dev_loss.append(dev_loss)
                self.train_acc_real.append(train_acc_real)
                self.train_acc_fake.append(train_acc_fake)
                self.dev_acc_real.append(dev_acc_real)
                self.dev_acc_fake.append(dev_acc_fake)

            def store_sample(self, dev_fakes):
                self.dev_fakes.append(dev_fakes)

            def store_first_epoch_sample(self, dev_fakes):
                self.dev_fakes_first_epoch = dev_fakes

            def set_images(self, dev_images, dev_sketches):
                self.dev_images = dev_images
                self.dev_sketches = dev_sketches

            def plot_examples(self, n_columns=6, save=False, file_identifier=''):
                if file_identifier == '':
                    file_identifier = self.model_name
                sketches_row = np.concatenate(
                    [np.concatenate([sketch, sketch, sketch], axis=2).astype(float)
                        for sketch in self.dev_sketches[:n_columns]],
                    axis=1)
                originals_row = np.concatenate(self.dev_images[:n_columns], axis=1)
                first_row = np.concatenate([self.dev_fakes_first_epoch[i] for i in range(n_columns)], axis=1)
                training_rows = [np.concatenate(pics_list[:n_columns], axis=1) for pics_list in self.dev_fakes[1:]]
                display_rows = [sketches_row] + [first_row] + training_rows + [originals_row]
                full_grid = np.concatenate(display_rows, axis=0)
                if save:
                    sp.misc.imsave(file_identifier + '_plot_examples.jpg', full_grid)
                plt.figure(
                    figsize=(n_columns*3, len(display_rows)*3)
                )
                plt.imshow(full_grid)
                plt.show()

            def save_examples(self, file_identifier=''):
                if file_identifier == '':
                    file_identifier = self.model_name
                all_examples = self.dev_fakes_first_epoch + self.dev_fakes[1:]
                with open(file_identifier + '_examples.pickle', 'wb') as img_file:
                    pickle.dump(all_examples, img_file, protocol=pickle.HIGHEST_PROTOCOL)

            def plot_metrics(self):  # TODO
                pass

            def return_metrics(self, round=True):
                metrics_df = pd.DataFrame([
                    self.checkpoints,
                    self.train_loss,
                    self.dev_loss,
                    self.train_acc_real,
                    self.train_acc_fake,
                    self.dev_acc_real,
                    self.dev_acc_fake,
                ]).T
                metrics_df.columns = [
                    'checkpoint',
                    'cGAN train_loss',
                    'cGAN dev_loss',
                    'disc_train_acc_real',
                    'disc_train_acc_fake',
                    'disc_dev_acc_real',
                    'disc_dev_acc_fake',
                ]
                metrics_df['disc_mean_train_acc'] = \
                    (metrics_df['disc_train_acc_real'] + metrics_df['disc_train_acc_fake']) / 2
                metrics_df['disc_mean_dev_acc'] = \
                    (metrics_df['disc_dev_acc_real'] + metrics_df['disc_dev_acc_fake']) / 2
                if round:
                    metrics_df = metrics_df.round(2)
                # reorder columns so that mean discriminator accuracies (best
                # overall indicators) come first
                metrics_df = metrics_df[[
                    'checkpoint',
                    'cGAN train_loss',
                    'cGAN dev_loss',
                    'disc_mean_train_acc',
                    'disc_mean_dev_acc',
                    'disc_train_acc_real',
                    'disc_train_acc_fake',
                    'disc_dev_acc_real',
                    'disc_dev_acc_fake',
                ]]

                return metrics_df

            def save_metrics(self, file_identifier=''):
                if file_identifier == '':
                    file_identifier = self.model_name
                metrics_df = pd.DataFrame([
                    self.checkpoints,
                    self.train_loss,
                    self.dev_loss,
                    self.train_acc_real,
                    self.train_acc_fake,
                    self.dev_acc_real,
                    self.dev_acc_fake,
                ]).T
                metrics_df.columns = [
                    'checkpoint',
                    'train_loss',
                    'dev_loss',
                    'train_acc_real',
                    'train_acc_fake',
                    'dev_acc_real',
                    'dev_acc_fake',
                ]

                metrics_df['disc_mean_train_acc'] = \
                    (metrics_df['disc_train_acc_real'] + metrics_df['disc_train_acc_fake']) / 2
                metrics_df['disc_mean_dev_acc'] = \
                    (metrics_df['disc_dev_acc_real'] + metrics_df['disc_dev_acc_fake']) / 2
                # reorder columns so that mean discriminator accuracies (best
                # overall indicators) come first
                metrics_df = metrics_df[[
                    'checkpoint',
                    'cGAN train_loss',
                    'cGAN dev_loss',
                    'disc_mean_train_acc',
                    'disc_mean_dev_acc',
                    'disc_train_acc_real',
                    'disc_train_acc_fake',
                    'disc_dev_acc_real',
                    'disc_dev_acc_fake',
                ]]

                metrics_df.to_csv(file_identifier+'_metrics.csv', index=False)

        self.history = History(model_name=self.name)


        # define model-building functions
        def encoding_layer(layer_input, filter_count, batch_norm=True, gauss_init=True):
            if gauss_init:
                weights_init = tf.keras.initializers.RandomNormal(mean=0, stddev=0.05)
            else:  # keras default: Glorot (aka Xavier) uniform initializer
                weights_init = 'glorot_uniform'
            layer = layers.Conv2D(filter_count, kernel_size=4, strides=2, \
                    padding='same', kernel_initializer=weights_init)(layer_input)
            layer = layers.LeakyReLU(alpha=0.2)(layer)
            if batch_norm:
                layer = layers.BatchNormalization()(layer)
            return layer

        def decoding_layer(layer_input, concat_layer, filter_count=512, dropout=False, gauss_init=True):
            if gauss_init:
                weights_init = tf.keras.initializers.RandomNormal(mean=0, stddev=0.05)
            else:  # keras default: Glorot (aka Xavier) uniform initializer
                weights_init = 'glorot_uniform'
            layer = layers.UpSampling2D(size=2)(layer_input)
            layer = layers.Conv2D(filter_count, kernel_size=4, strides=1, padding='same', \
                                activation='relu', kernel_initializer=weights_init)(layer)
            layer = layers.BatchNormalization()(layer)
            if dropout:
                layer = layers.Dropout(rate=0.5)(layer)
            layer = layers.Concatenate()([layer, concat_layer])
            return layer

        def build_generator(num_output_channels=3, final_activation='tanh', gauss_init=True):
            inputs = layers.Input(shape=(128,128,1))

            # encoder
            en_1 = encoding_layer(inputs, 64, batch_norm=False, gauss_init=gauss_init)  # shape = 64 x 64 x 64
            en_2 = encoding_layer(en_1, 128, gauss_init=gauss_init)  # shape = 32 x 32 x 128
            en_3 = encoding_layer(en_2, 256, gauss_init=gauss_init)  # shape = 16 x 16 x 256
            en_4 = encoding_layer(en_3, 512, gauss_init=gauss_init)  # shape = 8 x 8 x 512
            en_5 = encoding_layer(en_4, 512, gauss_init=gauss_init)  # shape = 4 x 4 x 512
            en_6 = encoding_layer(en_5, 512, gauss_init=gauss_init)  # shape = 2 x 2 x 512
            en_7 = encoding_layer(en_6, 512, gauss_init=gauss_init)  # shape = 1 x 1 x 512

            # decoder
            de_1 = decoding_layer(en_7, en_6, 512, dropout=True, gauss_init=gauss_init)  # shape = 2 x 2 x 1024
            de_2 = decoding_layer(de_1, en_5, 512, dropout=True, gauss_init=gauss_init)  # shape = 4 x 4 x 1024
            de_3 = decoding_layer(de_2, en_4, 512, dropout=True, gauss_init=gauss_init)  # shape = 8 x 8 x 1024
            de_4 = decoding_layer(de_3, en_3, 512, gauss_init=gauss_init)  # shape = 16 x 16 x 1024
            de_5 = decoding_layer(de_4, en_2, 256, gauss_init=gauss_init)  # shape = 32 x 32 x 512
            de_6 = decoding_layer(de_5, en_1, 128, gauss_init=gauss_init)  # shape = 64 x 64 x 256
            new_image = layers.UpSampling2D(size=2)(de_6)  # shape = 128 x 128 x 256
            new_image = layers.Conv2D(
                num_output_channels,
                kernel_size=4,
                strides=1,
                padding='same',
                kernel_initializer = tf.keras.initializers.RandomNormal(mean=0, stddev=0.05) if gauss_init else 'glorot_uniform',
                activation=final_activation
            )(new_image)  # shape = 128 x 128 x num_output_channels

            return Model(inputs, new_image)

        def build_discriminator(input_channels=4, gauss_init=True):
            '''input_channels = 6 for look-at-both discriminator, 4 for look-at-each-with-sketch discriminator'''
            inputs = layers.Input(shape=(128, 128, input_channels))

            en_1 = encoding_layer(inputs, 64, batch_norm=False, gauss_init=gauss_init) # shape = 64 x 64 x 64
            en_2 = encoding_layer(en_1, 128, gauss_init=gauss_init)  # shape = 32 x 32 x 128
            en_3 = encoding_layer(en_2, 256, gauss_init=gauss_init)  # shape = 16 x 16 x 256
            patches = layers.Conv2D(1,
                                    kernel_size=4,
                                    strides=1,
                                    padding='valid',
                                    kernel_initializer = tf.keras.initializers.RandomNormal(mean=0, stddev=0.05) if gauss_init else 'glorot_uniform',
                                    activation='sigmoid'
                                   )(en_3) # shape = 13 x 13 x 1
            prediction = layers.AveragePooling2D(pool_size=13)(patches)

            return Model(inputs, prediction)

        def build_dual_discriminator(single_discriminator):
            concat_real_input = layers.Input(shape=(128,128,4))
            concat_fake_input = layers.Input(shape=(128,128,4))
            pred_on_real = single_discriminator(concat_real_input)
            pred_on_fake = single_discriminator(concat_fake_input)
            return Model(
                [concat_real_input, concat_fake_input],
                [pred_on_real, pred_on_fake]
            )


        # define loss functions for models
        def disc_loss_real(y_true, y_pred):
            ''' like cGAN loss log term, but negative because the discrim. aims to maximize'''
            return -1 * tf.log(y_pred + 1e-10) # slight bias for numerical stability

        def disc_loss_fake(y_true, y_pred):
            ''' like cGAN loss log term, but negative because the discrim. aims to maximize'''
            return -1 * tf.log(1 - y_pred + 1e-10) # slight bias for numerical stability

        def cGAN_loss_photo(real_photo_fives, fake_photos):
            return tf.reduce_mean(losses.mean_absolute_error(real_photo_fives, fake_photos)) \
                    * self.lambda_

        def cGAN_prediction_loss(y_true, all_predictions):
            # previous_best logs the (0-->4) indices of the previous top-scoring fakes

            # prohibit model from choosing either of the past two top-scorers
            real_pred = all_predictions[5]
            eligible_preds = tf.concat([all_predictions[i] for i in range(5)
                                        if i not in self.previous_best[-2:]],
                                        -1)
            max_eligible_fake_pred = tf.reduce_max(eligible_preds)
            max_pred_index = tf.argmax(all_predictions[:5])

            # update log of indices of previous best fakes
            del self.previous_best[0]
            self.previous_best.append(max_pred_index)

            return tf.log(real_pred) + tf.log(1 - max_eligible_fake_pred)

        # create discriminator models (one as a component in the cGAN, and one
        # to train separataly). They use the same base discriminator so that the
        # weights are linked, but the component discriminator can't train them.
        self.single_discriminator = build_discriminator()
        self.dual_discriminator = build_dual_discriminator(self.single_discriminator)
        self.dual_discriminator.compile(
            optimizer=tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5),
            loss=[disc_loss_real, disc_loss_fake],
            metrics=[tf.keras.metrics.binary_accuracy, tf.keras.metrics.binary_accuracy])

        # create cGAN out of a generator and a discriminator
        self.generator = build_generator(num_output_channels=15, final_activation=self.gen_activation)
        sketch_input = layers.Input(shape=(128,128,1))
        photo_input = layers.Input(shape=(128,128,3))
        # print(photo_input.output_shape)
        fake_photos = self.generator(sketch_input)
        fake_1 = layers.Lambda(lambda x:x[:,:,:,:3])(fake_photos)
        fake_2 = layers.Lambda(lambda x:x[:,:,:,3:6])(fake_photos)
        fake_3 = layers.Lambda(lambda x:x[:,:,:,6:9])(fake_photos)
        fake_4 = layers.Lambda(lambda x:x[:,:,:,9:12])(fake_photos)
        fake_5 = layers.Lambda(lambda x:x[:,:,:,12:])(fake_photos)
        concat_real = layers.Concatenate()([sketch_input, photo_input])
        concat_fake_1 = layers.Concatenate()([sketch_input, fake_1])
        concat_fake_2 = layers.Concatenate()([sketch_input, fake_2])
        concat_fake_3 = layers.Concatenate()([sketch_input, fake_3])
        concat_fake_4 = layers.Concatenate()([sketch_input, fake_4])
        concat_fake_5 = layers.Concatenate()([sketch_input, fake_5])
        prediction_real = self.single_discriminator(concat_real)
        prediction_1 = self.single_discriminator(concat_fake_1)
        prediction_2 = self.single_discriminator(concat_fake_2)
        prediction_3 = self.single_discriminator(concat_fake_3)
        prediction_4 = self.single_discriminator(concat_fake_4)
        prediction_5 = self.single_discriminator(concat_fake_5)
        all_predictions = layers.Concatenate()([prediction_1, prediction_2, prediction_3, \
                                        prediction_4, prediction_5, prediction_real])
        all_predictions = layers.Flatten()(all_predictions)
        print(all_predictions.shape)
        self.cGAN = Model(
            [sketch_input, photo_input],
            [fake_photos, all_predictions]
        )
        self.cGAN.compile(
            optimizer=tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5),
            loss=[cGAN_loss_photo, cGAN_prediction_loss]
        )  # we don't need metrics because we only need the losses back from this one
        print('compiled generator')

    def train(self, img_pairs, dev_pairs, batch_size=4, epochs=15, \
              checkpoints_per_epoch=4, checkpoints_batch_size=20):

        # function to recenter image pixels around 0
        # i.e. change (max,min) from (0,1) to (-1,1)
        def recenter_pixels(img):
            return (img.astype('float32') * 2) - 1
        # function to un-re-center image pixels
        def unrecenter_pixels(img):
            return (img + 1) / 2

        def get_batches(img_pairs, batch_size):
            num_full_batches = len(img_pairs)//batch_size
            batch_indices = [num*batch_size for num in range(num_full_batches)]
            pair_batches = [ img_pairs[i:i+batch_size] for i in batch_indices ]
            if len(img_pairs) % batch_size != 0:
                pair_batches.append(img_pairs[batch_size*num_full_batches:])
            batches = [
                (
                    np.array([img_pair[0] for img_pair in pair_batch]),
                    np.array([np.reshape(img_pair[1], (128,128,1)) for img_pair in pair_batch])
                )
                for pair_batch in pair_batches
            ]
            return batches

        def perform_batch_train(batch):
            sketches = batch[1]
            true_photos = batch[0]
            true_photos_fives = tf.concat([true_photos,true_photos,true_photos,true_photos,true_photos],
                                            axis=-1)
            weights_before_train = self.single_discriminator.get_weights()
            cGAN_metrics = self.cGAN.train_on_batch(
                [sketches, true_photos],
                [true_photos_fives, np.zeros((true_photos.shape[0], 6))])
            # adjust discriminator weights to move two steps in the OPPOSITE direction
            # from the way they shifted with cGAN backprop (net move of one step)
            self.single_discriminator.set_weights(list(
                2 * np.array(weights_before_train) - np.array(self.single_discriminator.get_weights())
                ))

        def get_batch_metrics(batch, return_fakes=False):
            true_photos = batch[0]
            sketches = batch[1]
            fake_photosets = self.generator.predict_on_batch(sketches)
            # get only the top photo from those photos
            num_photos = fake_photosets.shape[0]
            # tup 0 = which photoset, tup[1] = index of best photo in that set
            fake_photos_best = tf.concat([ fake_photosets[tup[0],:,:,tup[1]:tup[1]+3]
                                for tup in zip(self.previous_best[-num_photos:],
                                                range(0,num_photos*3, 3)) ],
                                                axis=0)
            real_concats = np.concatenate([sketches, true_photos], axis=-1)
            fake_concats = np.concatenate([sketches, fake_photos_best], axis=-1)
            cGAN_metrics = self.cGAN.test_on_batch(
                [sketches, true_photos],
                [true_photos, np.zeros((num_photos, 5))])
            discriminator_metrics = self.dual_discriminator.test_on_batch(
                [real_concats, fake_concats],
                [np.ones((checkpoints_batch_size,1,1,1)), np.zeros((checkpoints_batch_size,1,1,1))]
            )
            if return_fakes:
                return cGAN_metrics, discriminator_metrics, fake_photos_best
            return cGAN_metrics, discriminator_metrics

        def log_metrics(train_batch, dev_batch, checkpoint, return_fakes=True):
            cGAN_train_metrics, dis_train_metrics = get_batch_metrics(train_batch)
            cGAN_dev_metrics, dis_dev_metrics, dev_fakes = get_batch_metrics(dev_batch, return_fakes=True)
            train_loss = cGAN_train_metrics[0] + cGAN_train_metrics[1]
            dev_loss = cGAN_dev_metrics[0] + cGAN_dev_metrics[1]
            # note: the indices below are rather odd and arbitrary; to view internal
            # labels call cGAN.dual_discriminator.metrics_names, and read the docs on
            # tf.keras.model.test_on_batch() return values
            self.history.update(
                checkpoint,
                train_loss,
                dev_loss,
                dis_train_metrics[3], # true_pred accuracy
                dis_train_metrics[5], # fake_pred accuracy
                dis_dev_metrics[3], # true_pred accuracy
                dis_dev_metrics[5], # fake_pred accuracy
            )
            if return_fakes:
                return dev_fakes

        # set dev images in history before recentering
        self.history.set_images(
            [pair[0] for pair in dev_pairs[:checkpoints_batch_size]],
            [np.reshape(pair[1], (128,128,1)) for pair in dev_pairs[:checkpoints_batch_size]]
        )

        # recenter pixel values to (-1,1) and split batches
        img_pairs = [ (recenter_pixels(pair[0]), recenter_pixels(pair[1]) )
                    for pair in img_pairs ]
        dev_pairs = [ (recenter_pixels(pair[0]), recenter_pixels(pair[1]) )
                    for pair in dev_pairs[:checkpoints_batch_size] ]

        # create batches
        batches = get_batches(img_pairs, batch_size)

        # create metric logging checkpoints
        log_batch_interval = len(batches) // checkpoints_per_epoch
        log_checkpoints = [0] + [
            (i * log_batch_interval) - 1
            for i in range(1, checkpoints_per_epoch+1)
        ]

        # training
        start = time()
        for epoch_num in range(epochs):
            print('\rRunning epoch {} of {}...             \n'.format(epoch_num+1, epochs), end='')
            if epoch_num == 1:
                log_checkpoints = log_checkpoints[1:] # only checks 0th step on the first epoch
            for batch_num, batch in enumerate(batches):
                print('\r  Training on batch {} of {}'.format(batch_num, len(batches)), end='')
                perform_batch_train(batch)

                # if at checkpoint, get batches and log metrics
                if batch_num in log_checkpoints:
                    if batch_num == 0:
                        start_index = 0
                        end_index = start_index+checkpoints_batch_size
                        checkpoint = 0
                    else:
                        start_index = max(
                            (batch_num + 1) * batch_size - (checkpoints_batch_size + 1),
                            0 )
                        end_index = min(start_index+checkpoints_batch_size, len(img_pairs)-1)
                        checkpoint = epoch_num + round((start_index + checkpoints_batch_size) / len(img_pairs), 2)
                    train_batch = get_batches(
                        img_pairs[start_index:end_index],
                        checkpoints_batch_size)[0]
                    dev_batch = get_batches(dev_pairs[:checkpoints_batch_size], checkpoints_batch_size)[0]
                    # log metrics
                    dev_fakes = log_metrics(train_batch, dev_batch, checkpoint, return_fakes=True)
                    dev_fakes = [unrecenter_pixels(fake) for fake in dev_fakes]
                    if epoch_num == 0:
                        self.history.store_first_epoch_sample(dev_fakes)
                    if batch_num == log_checkpoints[-1]:
                        self.history.store_sample(dev_fakes)

        minutes = round((time()-start)/60)
        print('\rTraining completed in {} minutes             '.format(minutes))

    def images_from_sketch(self, sketch, plot=False):
        # function to recenter image pixels around 0
        # i.e. change (max,min) from (0,1) to (-1,1)
        def recenter_pixels(img):
            return (img.astype('float32') * 2) - 1
        # function to un-re-center image pixels
        def unrecenter_pixels(img):
            return (img + 1) / 2

        fake_photoset = unrecenter_pixels(
            self.generator.predict_on_batch(
            np.reshape(recenter_pixels(sketch), (1,128,128,1))
            )[0]
        )
        fake_photos = [fake_photoset[i] for i in range(5)]

        if plot:
            plt.imshow(np.concat(fake_photos, axis=1))
            plt.show()

        return fake_photos



    def save_weights(self, filename_base=''):
        if filename_base == '':
            filename_base = self.name
        self.single_discriminator.save_weights(filename_base + '_disc_weights.h5')
        self.cGAN.save_weights(filename_base + '_cGAN_weights.h5')

    def load_weights(self, filename_base):
        self.single_discriminator.load_weights(filename_base + '_disc_weights.h5')
        self.cGAN.load_weights(filename_base + '_cGAN_weights.h5')
