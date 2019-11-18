import keras
import numpy as np
from math import ceil, floor, log
from utils import _read, weighted_loss, weighted_log_loss_metric
from albumentations import (
    Compose, HorizontalFlip, CLAHE, HueSaturationValue,
    RandomBrightness, RandomContrast, RandomGamma,
    ToFloat, ShiftScaleRotate, VerticalFlip
)
import efficientnet.keras as efn
# from resnext import ResNeXtImageNet
from tta_wrapper import tta_segmentation

from classification_models.keras import Classifiers
from keras.models import Model

test_images_dir = '/home/MARQNET/shared/rsna-intracranial-hemorrhage-detection/stage_2_test_images/'
train_images_dir = '/home/MARQNET/shared/rsna-intracranial-hemorrhage-detection/stage_2_train_images/'

class PredictionCheckpoint(keras.callbacks.Callback):

    def __init__(self, test_df, valid_df,
                 test_images_dir=test_images_dir,
                 valid_images_dir=train_images_dir,
                 batch_size=250, input_size=(512, 512, 5)):
        self.test_df = test_df
        self.valid_df = valid_df
        self.test_images_dir = test_images_dir
        self.valid_images_dir = valid_images_dir
        self.batch_size = batch_size
        self.input_size = input_size
        self.lr_annealing_counter = 0

    def on_train_begin(self, logs={}):
        self.test_predictions = []
        self.valid_predictions = []
        # self.valid_losses = []

    # def reduce_learnrate_custom(self, schedule='step', patience=3, direction='min', factor=0.4, min_delta=0.001, min_lr=1e-9):
    #     if len(self.valid_losses) < patience:
    #         return
    #     # eval = self.valid_losses[-int(patience):]
    #     # diff = [j - i for i, j in zip(eval[:-1], eval[1:])]
    #
    #     if direction == "max":
    #         if self.valid_losses[-1] <= self.valid_losses[(-1 - patience)]:
    #             new_lr = float(keras.backend.get_value(self.model.optimizer.lr)) * factor
    #             if new_lr < min_lr:
    #                 return
    #             else:
    #                 keras.backend.set_value(self.model.optimizer.lr, new_lr)
    #                 print('Plateauing ... ... Setting new learn rate: ' + str(
    #                     float(keras.backend.get_value(self.model.optimizer.lr))))
    #         else:
    #             return
    #     elif direction == "min":
    #         if self.valid_losses[-1] >= self.valid_losses[(-1 - patience)]:
    #             new_lr = float(keras.backend.get_value(self.model.optimizer.lr)) * factor
    #             if new_lr < min_lr:
    #                 return
    #             else:
    #                 keras.backend.set_value(self.model.optimizer.lr, new_lr)
    #                 print('Plateauing ... ... Setting new learn rate: ' + str(
    #                     float(keras.backend.get_value(self.model.optimizer.lr))))
    #         else:
    #             return


    def on_epoch_end(self, batch, logs={}):
        model = self.model

        # self.reduce_learnrate_custom()

        # model = tta_segmentation(self.model, h_flip=True, v_flip=True, h_shift=(-10, 10), v_shift=(-10, 10), merge='mean')
        self.test_predictions.append(
            model.predict_generator(
                DataGenerator(self.test_df.index,None, None, self.batch_size, self.input_size, self.test_images_dir),
                verbose=1)[:len(self.test_df)])
        print('Validating...')
        print(self.input_size)
        self.valid_predictions.append(
            model.predict_generator(
                DataGenerator(self.valid_df.index, None,None, self.batch_size, self.input_size, self.valid_images_dir), verbose=1)[:len(self.valid_df)])

        val_loss = weighted_log_loss_metric(self.valid_df.values,
                                   np.average(self.valid_predictions, axis=0,
                                              weights=[2**i for i in range(len(self.valid_predictions))]))
        # self.valid_losses.append(val_loss)
        print("validation loss: %.4f" % val_loss)

    def on_epoch_begin(self, batch, logs={}):
        lr = float(keras.backend.get_value(self.model.optimizer.lr))
        print('Using learn rate: '+str(lr))

    # def on_train_end(self, logs={}):
    #     print('Predicting with latest model...')
    #     self.test_predictions.append(
    #         self.model.predict_generator(
    #             DataGenerator(self.test_df.index, None, None, self.batch_size, self.input_size, self.test_images_dir),
    #             verbose=1)[:len(self.test_df)])

class DataGenerator(keras.utils.Sequence):

    def __init__(self, list_IDs, augmentations, labels=None, batch_size=1, img_size=(512, 512, 1),
                 img_dir=train_images_dir, *args, **kwargs):

        self.list_IDs = list_IDs
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.img_dir = img_dir
        self.on_epoch_end()
        self.augmentations = augmentations

    def __len__(self):
        return int(ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indices]

        if self.labels is not None:
            X, Y = self.__data_generation(list_IDs_temp)
            return X, Y
        else:
            X = self.__data_generation(list_IDs_temp)
            return X

    def on_epoch_end(self):
        if self.labels is not None:
            keep_prob = self.labels.iloc[:, 0].map({0: 0.35, 1: 0.5})
            keep = (keep_prob > np.random.rand(len(keep_prob)))
            self.indices = np.arange(len(self.list_IDs))[keep]
            np.random.shuffle(self.indices)
        else:
            self.indices = np.arange(len(self.list_IDs))

    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size, *self.img_size))

        if self.labels is not None:  # training phase
            Y = np.empty((self.batch_size, 6), dtype=np.float32)

            for i, ID in enumerate(list_IDs_temp):
                img = _read(self.img_dir + ID + ".dcm", self.img_size)
                if self.augmentations:
                    img = np.float32(img)
                    augmented = self.augmentations(image=img)
                    X[i,] = augmented['image']
                Y[i,] = self.labels.loc[ID].values

            return X, Y

        else:  # test phase
            for i, ID in enumerate(list_IDs_temp):
                X[i,] = _read(self.img_dir + ID + ".dcm", self.img_size)

            return X


class MyDeepModel:

    def __init__(self, engine, input_dims, batch_size=5, num_epochs=4, learning_rate=1e-3,
                 decay_rate=1.0, decay_steps=1, weights="imagenet", verbose=1):
        self.engine = engine
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.weights = weights
        self.verbose = verbose
        self._build()

    def create_effnet_model(self):
        base_model = efn.EfficientNetB0(weights='imagenet', include_top=False, pooling='avg',
                                        input_shape=(self.input_dims[0], self.input_dims[0], 3))
        x = base_model.output
        x = keras.layers.Dropout(0.05)(x)
        out = keras.layers.Dense(6, activation="sigmoid", name='dense_output')(x)

        return keras.models.Model(inputs=base_model.input, outputs=out)

    def create_resnet34_model(self):
        ResNet34, preprocess_input = Classifiers.get('resnet34')
        base_model = ResNet34((self.input_dims[0], self.input_dims[0], 3), weights='imagenet', include_top=False)
        x = keras.layers.GlobalAveragePooling2D()(base_model.output)
        out = keras.layers.Dense(6, activation='sigmoid', name='dense_output')(x)

        return keras.models.Model(inputs=base_model.input, outputs=out)

    def _build(self):
        #efficient nets custom
        if self.engine == 'effnet':
            self.model = self.create_effnet_model()

        elif self.engine == 'resnet_34':
            self.model = self.create_resnet34_model()

        elif self.engine == 'graynet':
            self.model = self.create_graynet_model()

        else:
            engine = self.engine(include_top=False, weights=self.weights, input_shape=self.input_dims,
                                 backend=keras.backend, layers=keras.layers,
                                 models=keras.models, utils=keras.utils)

            x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(engine.output)
            out = keras.layers.Dense(6, activation="sigmoid", name='dense_output')(x)

            self.model = keras.models.Model(inputs=engine.input, outputs=out)

        self.model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(self.learning_rate), metrics=[weighted_loss])

    def fit_and_predict(self, train_df, valid_df, test_df):
        # callbacks
        pred_history = PredictionCheckpoint(test_df, valid_df, input_size=self.input_dims)
        # checkpointer = keras.callbacks.ModelCheckpoint(filepath='%s-{epoch:02d}.hdf5' % self.engine.__name__, verbose=1, save_weights_only=True, save_best_only=False)



        scheduler = keras.callbacks.LearningRateScheduler(
            lambda epoch: self.learning_rate * pow(self.decay_rate, floor(epoch / self.decay_steps)))

        AUGMENTATIONS_TRAIN = Compose([
            HorizontalFlip(p=0.5),
            # VerticalFlip(p=0s.5),
            # RandomContrast(p=0.5),
            # RandomBrightness(p=0.5),
            # ShiftScaleRotate(),
        ])

        self.model.fit_generator(
            DataGenerator(
                train_df.index,
                AUGMENTATIONS_TRAIN,
                train_df,
                self.batch_size,
                self.input_dims,
                train_images_dir
            ),
            epochs=self.num_epochs,
            verbose=self.verbose,
            use_multiprocessing=True,
            workers=4,
            callbacks=[pred_history, scheduler]
        )

        return pred_history

    def save(self, path):
        self.model.save_weights(path)

    def load(self, path):
        self.model.load_weights(path)