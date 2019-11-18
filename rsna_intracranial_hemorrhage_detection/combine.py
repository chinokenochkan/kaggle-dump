import pandas as pd
import numpy as np

df_incep_fold1 = pd.read_csv('~/desktop/submission433.csv')
# df_incep_fold2 = pd.read_csv('~/desktop/submission_resnet_2folds.csv')
# df_resnet =pd.read_csv('~/desktop/submission3.csv')
final_label = df_incep_fold1.Label.round(1)
print(final_label.dtype)

print(df_incep_fold1.ID.head)
data = {
    'ID':df_incep_fold1.ID,
    'Label':final_label
}
test_df = pd.DataFrame(data)
print('Conversion complete...')
print(test_df.head)
test_df.to_csv('~/desktop/submission.csv', index=False)
