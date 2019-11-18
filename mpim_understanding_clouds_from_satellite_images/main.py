from cloud_images_segmentation_utillity_script import *
from keras.models import load_model
from sklearn.model_selection import ShuffleSplit
from tta_wrapper import tta_segmentation

warnings.filterwarnings("ignore")
TRAIN = True
ensb = 1

train = pd.read_csv('./data/understanding_cloud_organization/train.csv')
submission = pd.read_csv('./data/understanding_cloud_organization/sample_submission.csv')

# Preprocecss data
train['image'] = train['Image_Label'].apply(lambda x: x.split('_')[0])
train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[1])
submission['image'] = submission['Image_Label'].apply(lambda x: x.split('_')[0])
test = pd.DataFrame(submission['image'].unique(), columns=['image'])

# Create one column for each mask
train_df = pd.pivot_table(train, index=['image'], values=['EncodedPixels'], columns=['label'], aggfunc=np.min).reset_index()
train_df.columns = ['image', 'Fish_mask', 'Flower_mask', 'Gravel_mask', 'Sugar_mask']

print('Compete set samples:', len(train_df))
print('Test samples:', len(submission))

splits = 5
K = 5
# train set (90%) and validation set (10%)
ss = ShuffleSplit(n_splits=splits, test_size=0.2).split(train_df.index)

for k in range(K):

    print('fold: '+str(k+1))

    train_idx, val_idx = next(ss)

    X_train, X_val = train_df.iloc[train_idx], train_df.iloc[val_idx]

    # X_train, X_val = train_test_split(train_df, test_size=0.2)
    X_train['set'] = 'train'
    X_val['set'] = 'validation'
    test['set'] = 'test'

    print('Train samples: ', len(X_train))
    print('Validation samples: ', len(X_val))

    NOTES = '480x576_fold'
    MODEL = 'Unet'
    BACKBONE = 'seresnet18'
    BATCH_SIZE = 16
    EPOCHS = 40
    LEARNING_RATE = 1e-3
    HEIGHT = 480
    # WIDTH = 480
    WIDTH = 576
    CHANNELS = 3
    N_CLASSES = 4
    ES_PATIENCE = 5
    RLROP_PATIENCE = 3
    DECAY_DROP = 0.2
    model_path = MODEL+'_%s_%sx%s.h5' % (BACKBONE, HEIGHT, WIDTH)

    preprocessing = sm.get_preprocessing(BACKBONE)

    augmentation = albu.Compose([
                                 albu.HorizontalFlip(p=0.5),
                                 albu.VerticalFlip(p=0.5),
                                 albu.GridDistortion(p=0.5),
                                 albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0,
                                                       shift_limit=0.1, border_mode=0, p=0.5)
                                ])

    train_base_path = './data/understanding_cloud_organization/train_images/'
    test_base_path = './data/understanding_cloud_organization/test_images/'
    train_images_dest_path = 'base_dir/train_images/'
    validation_images_dest_path = 'base_dir/validation_images/'
    test_images_dest_path = 'base_dir/test_images/'

    # Making sure directories don't exist
    if os.path.exists(train_images_dest_path):
        shutil.rmtree(train_images_dest_path)
    if os.path.exists(validation_images_dest_path):
        shutil.rmtree(validation_images_dest_path)
    if os.path.exists(test_images_dest_path):
        shutil.rmtree(test_images_dest_path)

    # Creating train, validation and test directories
    os.makedirs(train_images_dest_path)
    os.makedirs(validation_images_dest_path)
    os.makedirs(test_images_dest_path)


    def preprocess_data(df, HEIGHT=HEIGHT, WIDTH=WIDTH):
        '''
        This function needs to be defined here, because it will be called with no arguments,
        and must have the default parameters from the beggining of the notebook (HEIGHT and WIDTH)
        '''
        df = df.reset_index()
        for i in range(df.shape[0]):
            item = df.iloc[i]
            image_id = item['image']
            item_set = item['set']
            if item_set == 'train':
                preprocess_image(image_id, train_base_path, train_images_dest_path, HEIGHT, WIDTH)
            if item_set == 'validation':
                preprocess_image(image_id, train_base_path, validation_images_dest_path, HEIGHT, WIDTH)
            if item_set == 'test':
                preprocess_image(image_id, test_base_path, test_images_dest_path, HEIGHT, WIDTH)


    # Pre-procecss train set
    pre_process_set(X_train, preprocess_data)

    # Pre-procecss validation set
    pre_process_set(X_val, preprocess_data)

    # Pre-procecss test set
    pre_process_set(test, preprocess_data)

    train_generator = DataGenerator(
                      directory=train_images_dest_path,
                      dataframe=X_train,
                      target_df=train,
                      batch_size=BATCH_SIZE,
                      target_size=(HEIGHT, WIDTH),
                      n_channels=CHANNELS,
                      n_classes=N_CLASSES,
                      preprocessing=preprocessing,
                      augmentation=augmentation,)

    valid_generator = DataGenerator(
                      directory=validation_images_dest_path,
                      dataframe=X_val,
                      target_df=train,
                      batch_size=BATCH_SIZE,
                      target_size=(HEIGHT, WIDTH),
                      n_channels=CHANNELS,
                      n_classes=N_CLASSES,
                      preprocessing=preprocessing,)

    if MODEL == 'Unet':
        model = sm.Unet(backbone_name=BACKBONE,
                        encoder_weights='imagenet',
                        classes=N_CLASSES,
                        activation='sigmoid',
                        input_shape=(HEIGHT, WIDTH, CHANNELS))
    elif MODEL == 'PSPNet':
        model = sm.PSPNet(backbone_name=BACKBONE,
                        encoder_weights='imagenet',
                        classes=N_CLASSES,
                        activation='sigmoid',
                        input_shape=(HEIGHT, WIDTH, CHANNELS))
    elif MODEL == 'FPN':
        model = sm.FPN(backbone_name=BACKBONE,
                        encoder_weights='imagenet',
                        classes=N_CLASSES,
                        activation='sigmoid',
                        input_shape=(HEIGHT, WIDTH, CHANNELS))

    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', mode='min', save_best_only=True, save_weights_only=True)
    es = EarlyStopping(monitor='val_loss', mode='min', patience=ES_PATIENCE, restore_best_weights=True, verbose=1)
    rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=RLROP_PATIENCE, factor=DECAY_DROP, min_lr=1e-6, verbose=1)

    metric_list = [dice_coef, sm.metrics.iou_score, get_kaggle_dice(pix=0.5, dim=(HEIGHT, WIDTH))]
    callback_list = [checkpoint, es, rlrop]
    optimizer = RAdam(learning_rate=LEARNING_RATE, warmup_proportion=0.1)

    model.compile(optimizer=optimizer, loss=sm.losses.bce_dice_loss, metrics=metric_list)
    model.summary()

    #############
    # Training  #
    #############

    STEP_SIZE_TRAIN = len(X_train)//BATCH_SIZE
    STEP_SIZE_VALID = len(X_val)//BATCH_SIZE

    if TRAIN:
        history = model.fit_generator(generator=train_generator,
                                      steps_per_epoch=STEP_SIZE_TRAIN,
                                      validation_data=valid_generator,
                                      validation_steps=STEP_SIZE_VALID,
                                      callbacks=callback_list,
                                      epochs=EPOCHS,
                                      verbose=2).history

    # model = load_model('./resnet18_trained_longer.h5', custom_objects={'RAdam':RAdam, 'binary_crossentropy_plus_dice_loss':sm.losses.bce_dice_loss, 'dice_coef':dice_coef, 'iou_score':sm.metrics.iou_score, 'f1-score':sm.metrics.f1_score})

    model = tta_segmentation(model, h_flip=True, v_flip=True, h_shift=(-10, 10), v_shift=(-10, 10), merge='mean')

    test_df = []; preds_ = []

    for i in range(0, test.shape[0], 300):
        batch_idx = list(range(i, min(test.shape[0], i + 300)))
        print('batch size: ' + str(len(batch_idx)))
        print('batch: '+str(batch_idx))
        batch_set = test[batch_idx[0]: batch_idx[-1] + 1]

        test_generator = DataGenerator(
            directory=test_images_dest_path,
            dataframe=batch_set,
            target_df=submission,
            batch_size=1,
            target_size=(HEIGHT, WIDTH),
            n_channels=CHANNELS,
            n_classes=N_CLASSES,
            preprocessing=preprocessing,
            mode='predict',
            shuffle=False)

        preds = model.predict_generator(test_generator)

        for index, b in enumerate(batch_idx):
            preds_.append(preds[index,])

    np.save('raw_preds_'+BACKBONE+'_'+str(ensb)+'_'+NOTES+str(k+1),preds_)