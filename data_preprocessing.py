import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from gc import collect
from pprint import pprint
from sklearn import set_config
from sklearn.preprocessing import MinMaxScaler

from utils import PrintColor

scaler = MinMaxScaler()

set_config(transform_output = "pandas")
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 50)

print()
collect()

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

train_df = pd.read_csv('data/train.csv')
plt.rcParams.update({'font.size': 16})


# Making sklearn pipeline outputs as dataframe:-
from sklearn import set_config
set_config(transform_output = "pandas")
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 50)

print()
collect()


print(train_df.info())
print(train_df.isna().sum())
print(train_df.describe())

# 取出价格列
prices = train_df.iloc[0:10].filter(regex = r"price$|p$").columns
PrintColor(f"\nPrice columns\n")
pprint(prices)

# 特征交叉
def MakeFtre(df: pd.DataFrame, prices: list, median_vol: pd.DataFrame) -> pd.DataFrame:
    """
    This function creates new features using the price columns. This was used in a baseline notebook as below-
    https://www.kaggle.com/code/yuanzhezhou/baseline-lgb-xgb-and-catboost

    Inputs-
    df:- pd.DataFrame -- input dataframe
    cols:- price columns for transformation
    median_vol:- pd.DataFrame -- stock wise median volume for train set

    Returns-
    df:- pd.DataFrame -- dataframe with extra columns
    """;

    features = ['overall_medvol', "first5min_medvol", "last5min_medvol",
                'seconds_in_bucket', 'imbalance_buy_sell_flag',
                'imbalance_size', 'matched_size', 'bid_size', 'ask_size',
                'reference_price', 'far_price', 'near_price', 'ask_price', 'bid_price', 'wap',
                'imb_s1', 'imb_s2'
                ];

    df = df.merge(median_vol, how='left', left_on="stock_id", right_index=True)

    print('after merge:------')
    print(df)

    df['imb_s1'] = df.eval('(bid_size-ask_size)/(bid_size+ask_size)').astype(np.float32)
    df['imb_s2'] = df.eval('(imbalance_size-matched_size)/(matched_size+imbalance_size)').astype(np.float32)

    for i, a in enumerate(prices):
        for j, b in enumerate(prices):
            if i > j:
                df[f'{a}_{b}_imb'] = df.eval(f'({a}-{b})/({a}+{b})')
                features.append(f'{a}_{b}_imb')

    for i, a in enumerate(prices):
        for j, b in enumerate(prices):
            for k, c in enumerate(prices):
                if i > j and j > k:
                    max_ = df[[a, b, c]].max(axis=1)
                    min_ = df[[a, b, c]].min(axis=1)
                    mid_ = df[[a, b, c]].sum(axis=1) - min_ - max_

                    df[f'{a}_{b}_{c}_imb2'] = ((max_ - mid_) / (mid_ - min_)).astype(np.float32)
                    features.append(f'{a}_{b}_{c}_imb2')

    return df


print()
collect()

# 数据预处理
def preprocess(df, mode='train'):
    ##########################
    # Transform
    ##########################
    df['matched_size'] = df['matched_size'].diff(2)
    df['imbalance_size'] = df['imbalance_size'].diff(2)
    df['bid_size'] = df['bid_size'].diff(2)
    df['ask_size'] = df['ask_size'].diff(2)

    ##########################
    # Feature Engineering
    ##########################
    df['log_return'] = np.log(df['wap'])
    for i in range(1, 11):
        df[f'imbalance_size_lag_{i}'] = df.groupby('stock_id')['imbalance_size'].shift(i)
        df[f'reference_price_lag_{i}'] = df.groupby('stock_id')['reference_price'].shift(i)
        df[f'matched_size_lag_{i}'] = df.groupby('stock_id')['matched_size'].shift(i)
        df[f'bid_price_lag_{i}'] = df.groupby('stock_id')['bid_price'].shift(i)
        df[f'ask_price_lag_{i}'] = df.groupby('stock_id')['ask_price'].shift(i)
        df[f'wap_{i}'] = df.groupby('stock_id')['wap'].shift(i)
    df['bid_size_lag_1'] = df.groupby('stock_id')['bid_size'].shift(1)
    df['ask_size_lag_1'] = df.groupby('stock_id')['ask_size'].shift(1)

    # Creating an output dataframe for the results:-
    median_vol = \
        pd.DataFrame(columns=['overall_medvol', "first5min_medvol", "last5min_medvol"],
                     index=range(0, 200, 1)
                     )
    # Creating the overall median volumes- this is from public work:-
    median_vol['overall_medvol'] = \
        df[['stock_id', "bid_size", "ask_size"]]. \
            groupby("stock_id")[["bid_size", "ask_size"]]. \
            median(). \
            sum(axis=1). \
            values.flatten()
    # Creating median volume with near and far price information:-
    median_vol['last5min_medvol'] = \
        df[['stock_id', "bid_size", "ask_size", "far_price", "near_price"]]. \
            dropna(). \
            groupby('stock_id')[["bid_size", "ask_size"]]. \
            median().sum(axis=1).values.flatten()
    # Creating median volume without near and far price information:-
    median_vol['first5min_medvol'] = \
        df.loc[(df['far_price'].isna()) | (df['near_price'].isna()),
        ['stock_id', "bid_size", "ask_size"]
        ]. \
            groupby('stock_id')[["bid_size", "ask_size"]]. \
            median().sum(axis=1).values.flatten()
    PrintColor(f"\n---> Median stock price information-  sample = 10 rows\n")
    print(median_vol.sample(10).style.format(formatter='{:,.2f}'))

    df = MakeFtre(df=df, prices=prices, median_vol=median_vol)
    print(df)
    print(df.info())
    print(df.columns.tolist())

    ##########################
    # Impute
    ##########################
    df = df.replace([np.inf, -np.inf], np.nan)  # 把无穷数值替换成空值
    df.fillna(0, inplace=True)

    ##########################
    # Normalize
    ##########################
    if mode == 'train':
        cols_to_norm = df.drop(['stock_id', 'time_id', 'date_id', 'row_id', 'target'], axis=1).columns
        print(np.isinf(df[cols_to_norm]).any())
        df[cols_to_norm] = scaler.fit_transform(df[cols_to_norm])
    elif mode == 'test':
        cols_to_norm = df.drop(['stock_id', 'date_id', 'row_id'], axis=1).columns
        df[cols_to_norm] = scaler.fit_transform(df[cols_to_norm])
    else:
        print('Wrong Mode.')
    return df

# train_df = preprocess(train_df)
# print(train_df)
#
# # 导出处理后的训练数据
# train_df.to_parquet(f"train.parquet")



