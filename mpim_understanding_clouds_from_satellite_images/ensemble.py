import numpy as np
import cv2
from cloud_images_segmentation_utillity_script import *

N_CLASSES=4

FILES = ['raw_preds_resnet18_1.npy','raw_preds_resnet34_1_no_hflip_vflip.npy', 'raw_preds_seresnet18_1.npy', 'raw_preds_resnet34_1_hflip_0.3_fpn.npy']
submission = pd.read_csv('./data/understanding_cloud_organization/sample_submission.csv')
submission['image'] = submission['Image_Label'].apply(lambda x: x.split('_')[0])
test = pd.DataFrame(submission['image'].unique(), columns=['image'])
test['set'] = 'test'

#############
# averaging #
#############

print('Averaging... ')
print('Using files: '+ str(FILES))

def load_and_resize(file):
    preds = np.load(file)
    print('array with shape: '+str(preds.shape))
    if preds.shape[1] != 384 or preds.shape[2] != 480:
        print('resizing...')
        nu = []
        for b in range(0, 3698):
            pred_masks_post = preds[b,].astype('float32')
            acc = []
            for class_index in range(N_CLASSES):
                acc.append(cv2.resize(pred_masks_post[..., class_index], (480, 384)))
            nu.append(acc)
        preds = np.array(nu)
        preds = preds.transpose(0, 2, 3, 1)
    print('processed, array shape: '+str(preds.shape))
    return preds

pred_1 = load_and_resize(FILES[0])
pred_2 = load_and_resize(FILES[1])

com = pred_1 + pred_2
pred_1 = None
pred_2 = None

pred_3 = load_and_resize(FILES[2])
pred_4 = load_and_resize(FILES[3])

# t = 0.5
# pred_1 = pred_1**t
# pred_2 = pred_2**t
# pred_3 = pred_3**t
# pred_4 = pred_4**t


preds = (pred_3 + pred_4 + com)/4.0


print('Averaged... New Shape: '+str(preds.shape))

test_df, N_CLASSES = [], 4

class_names = ['Fish  ', 'Flower', 'Gravel', 'Sugar ']
best_thresholds = [.5, .5, .5, .35]
# best_thresholds = [.6, .6, .6, .6]
best_masks = [25000, 20000, 22500, 15000]
# best_masks = [15000, 15000, 15000, 15000]

for index, name in enumerate(class_names):
    print('%s treshold=%.2f mask size=%d' % (name, best_thresholds[index], best_masks[index]))

for b in range(0, 3698):
    filename = test['image'].iloc[b]
    image_df = submission[submission['image'] == filename].copy()

    ### Post procecssing
    pred_masks_post = preds[b,].astype('float32')
    for class_index in range(N_CLASSES):
        pred_mask = pred_masks_post[..., class_index]
        pred_mask = post_process(pred_mask, threshold=best_thresholds[class_index], min_size=best_masks[class_index])
        pred_masks_post[..., class_index] = pred_mask

    pred_rles_post = build_rles(pred_masks_post, reshape=(350, 525))
    image_df['EncodedPixels_post'] = pred_rles_post
    test_df.append(image_df)

sub_df = pd.concat(test_df)
submission_df_post = sub_df[['Image_Label' ,'EncodedPixels_post']]
submission_df_post.columns = ['Image_Label' ,'EncodedPixels']
submission_df_post.to_csv('submission_post.csv', index=False)

