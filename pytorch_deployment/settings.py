import os
import torch.nn as nn
import torch



folder_path = '/Users/wbchoi/PycharmProjects/ai_projects/C_ITS/prediction_modeling'

labels = ['교통사고 N', '교통사고 Y']

data_dir = os.path.join(folder_path, 'full_data_kma_hours_lightgbm_feature_extract.csv')
model_path = os.path.join(folder_path, 'pytorch_DNN_model/pytorch_model_epoch_40.pkl')

PORT = 8000



model = nn.Sequential(nn.Linear(36, 64),
                      nn.ReLU(),
                      nn.BatchNorm1d(64),

                      nn.Linear(64, 64),
                      nn.ReLU(),
                      nn.Dropout(p=0.2),
                      nn.BatchNorm1d(64),

                      nn.Linear(64, 64),
                      nn.ReLU(),
                      nn.Dropout(p=0.2),
                      nn.BatchNorm1d(64),

                      nn.Linear(64, 64),
                      nn.ReLU(),
                      nn.Dropout(p=0.2),
                      nn.BatchNorm1d(64),

                      nn.Linear(64, 1)
                      )

step_size = 1
gamma = 1
weight_decay = 10e-4
lr = 10e-5

weight_ratio = 4203.3
class_weight = torch.FloatTensor([weight_ratio])


criterion = nn.BCEWithLogitsLoss(pos_weight = class_weight, reduction = 'mean')

optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=gamma, step_size=step_size, last_epoch=-1)