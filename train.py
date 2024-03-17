import math
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error
from torch.utils.data import DataLoader

from model import CNN, CNNLSTMModel, Transformer, CNNLSTMModel_ECA
from utils import get_dataset, correlation_coefficient_loss, PurgedGroupTimeSeriesSplit, set_seeds

train_df = pd.read_csv('data/train.csv')

class CFG():
    BATCH_SIZE = 32
    N_EPOCHS = 500
    LEARNING_RATE = 0.0001
    N_FOLDS = 5
    TARGET_COLS = ['target']
    SEED = 2023
    N_ASSETS = train_df['stock_id'].nunique()
    WEIGHT_DECAY = 0.01

CFG = CFG()
set_seeds(CFG.SEED)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available and being used")
else:
    device = torch.device("cpu")
    print("GPU is not available, using CPU instead")

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_score(y_val, y_pred, metrics):
    all_scores = {}
    for metric in metrics:
        metric_scores = []
        for i in range(len(y_val)):
            score = metric(y_val[i], y_pred[i])
            metric_scores.append(score)
        all_scores[str(metric.__name__)] = np.mean(metric_scores)
    return all_scores


def train_fn(fold, train_dataloader, model, loss_fn, optimizer, epoch, device):
    losses = AverageMeter()
    model.train()
    for step, (inputs, labels) in enumerate(train_dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        losses.update(loss.item(), labels.size(0))
        loss.backward()
        optimizer.step()
    return losses.avg


def valid_fn(valid_dataloader, model, loss_fn, device):
    losses = AverageMeter()
    model.eval()
    preds = []
    for step, (input_ids, labels) in enumerate(valid_dataloader):
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.no_grad():
            outputs = model(input_ids)
            loss = loss_fn(outputs, labels)
        losses.update(loss.item(), batch_size)
        preds.append(outputs.to('cpu').numpy())
    predictions = np.concatenate(preds)
    return losses.avg, predictions


def train_loop(train, val, fold):
    print(f'----------------- Fold: {fold + 1} -----------------')
    ##########################
    # Create Dataset
    ##########################
    train_ds = get_dataset(train)
    val_ds = get_dataset(val)
    _, y_val = val_ds[:]

    ##########################
    # DataLoader
    ##########################
    train_dataloader = DataLoader(
        train_ds,
        batch_size=CFG.BATCH_SIZE
    )
    val_dataloader = DataLoader(
        val_ds,
        batch_size=CFG.BATCH_SIZE
    )

    ##########################
    # Model
    ##########################
    model = CNN().to(device)
    # model = CNNLSTMModel().to(device)
    # model = CNNLSTMModel_ECA().to(device)
    # model = Transformer(feature_num=115, d_model=64, nhead=8, num_layers=1).to(device)

    ##########################
    # Optimizer
    ##########################
    optimizer_parameters = [{'params': [p for n, p in model.named_parameters()],
                             'lr': CFG.LEARNING_RATE,
                             'weight_decay': CFG.WEIGHT_DECAY
                             }, ]
    optimizer = torch.optim.AdamW(optimizer_parameters)

    ##########################
    # Trainig Parameters
    ##########################
    loss_fn = nn.MSELoss()
    min_mae_score = np.inf
    min_mse_score = np.inf
    min_rmse_score = np.inf
    wait = 0
    patience = 30
    metrics = [mean_absolute_error, correlation_coefficient_loss, r2_score, mean_squared_error]

    ##########################
    # Training Loop
    ##########################
    for epoch in range(CFG.N_EPOCHS):
        avg_loss = train_fn(fold, train_dataloader, model, loss_fn, optimizer, epoch, device)

        avg_val_loss, y_pred = valid_fn(val_dataloader, model, loss_fn, device)

        score = get_score(y_val, y_pred, metrics)

        print(
            f"Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f} avg_val_loss: {avg_val_loss:.4f} MAE: {score['mean_absolute_error']:.4f} R2: {score['r2_score']:.4f} Coefficient Correlation: {score['correlation_coefficient_loss']:.4f} MSE: {score['mean_squared_error']:.4f} RMSE: {math.sqrt(score['mean_squared_error']):.4f}")

        wait += 1
        if round(score['mean_absolute_error'], 4) < round(min_mae_score, 4):
            min_mae_score = score['mean_absolute_error']
            min_mse_score = score['mean_squared_error']
            min_rmse_score = math.sqrt(score['mean_squared_error'])
            wait = 0
            torch.save(model.state_dict(), f'model_fold_{fold + 1}')
            print(f'﹂ saving model with score: {min_mae_score:.4f}')
        if wait >= patience:
            print(f'Triggering Early Stopping on epoch {epoch + 1}')

            # 转换为一维向量
            vec_val = y_val.reshape(-1)
            vec_pred = y_pred.reshape(-1)
            print(vec_val)
            print(vec_pred)
            # 将一维向量转换为 DataFrame
            df1 = pd.DataFrame(vec_val, columns=['vec_val'])
            df2 = pd.DataFrame(vec_pred, columns=['vec_pred'])
            df = pd.concat([df1,df2],axis=1)
            # df = df[df.vec_val != 0]

            df.to_csv('out/vec_pred_fold_' + str(fold) + '.csv', index=False)
            # df2.to_csv('vec_pred_fold' + str(fold) + '.csv', index=False)

            return min_mae_score, min_mse_score, min_rmse_score

    gc.collect()


def train(df):
    Fold = PurgedGroupTimeSeriesSplit(n_splits=CFG.N_FOLDS,
                                      max_train_group_size=10000,
                                      max_val_group_size=200,
                                      val_group_gap=10)
    mae_scores = np.empty([CFG.N_FOLDS])
    mse_scores = np.empty([CFG.N_FOLDS])
    rmse_scores = np.empty([CFG.N_FOLDS])
    groups = df['time_id']
    for fold, (train_index, val_index) in enumerate(Fold.split(df, df[CFG.TARGET_COLS], groups=groups)):
        train = df.iloc[train_index].reset_index(drop=True)
        val = df.iloc[val_index].reset_index(drop=True)
        val.to_csv('out/val_' + 'fold_' + str(fold) + '.csv')
        min_mae_score, min_mse_score, min_rmse_score = train_loop(train, val, fold)
        mae_scores[fold] = min_mae_score
        mse_scores[fold] = min_mse_score
        rmse_scores[fold] = min_rmse_score
    print(
        f'Average MAE across folds: {np.mean(mae_scores)} Average MSE across folds: {np.mean(mse_scores)} Average RMSE across folds: {np.mean(rmse_scores)}')

# 读取处理后的训练数据并打印
train_df = pd.read_parquet(f"train.parquet")

print(train_df)

# train_df = train_df.iloc[0:20000]
# 开始训练
train(train_df)

def get_models():
    models = []
    for i in range(CFG.N_FOLDS):
        model = CNN()
        model.load_state_dict(torch.load(f'/kaggle/working/model_fold_{i+1}'))
        model.eval()
        models.append(model)
    return models