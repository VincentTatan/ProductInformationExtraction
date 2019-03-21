# import standard libraries
import os
import sys

# import libraries
import pandas as pd
import numpy as np
from PIL import Image

# np.set_printoptions(threshold=np.inf)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)

pathCwd = os.getcwd()
print(pathCwd)

# read train and test dataset for mobile only
df_Train_Full = pd.read_csv(f'{pathCwd}/ndsc-advanced/mobile_data_info_train_competition.csv')
print(df_Train_Full.head())
print(df_Train_Full.shape)
print(df_Train_Full.dtypes)

df_Test_Val = pd.read_csv(f'{pathCwd}/ndsc-advanced/mobile_data_info_val_competition.csv')
print(df_Test_Val.head())
print(df_Test_Val.shape)
print(df_Test_Val.dtypes)

# declare path for images
pathData = 'D:/NDSC_Dataset/'
pathOut = 'D:/NDSC_Dataset/mobile_csv'

# for enum, image_path in enumerate(df_Train_Full.head(2)['image_path']):
#     strPath = pathData + image_path
#     yay = Image.open(strPath)
#     yay.show()
#     yay.thumbnail((128, 128), Image.ANTIALIAS)
#     a = np.asarray(yay)
# #     np.savetxt(f"{enum}.csv", a, delimiter=",")
#     print(a)
#     print(a.shape)

# to iterate through later on
listYvarcolumns = ['Operating System', 'Features', 'Network Connections', 'Memory RAM', 'Brand', 'Warranty Period',
                  'Storage Capacity', 'Color Family', 'Phone Model', 'Camera', 'Phone Screen Size']

for colYvar in ['Brand']:  # listYvarcolumns
    print(colYvar)
    # now just use brand only
    print(f"before {colYvar} dropna", df_Train_Full.shape)
    df_Train_Brand = df_Train_Full.copy()
    df_Train_Brand.dropna(subset=[colYvar], inplace=True)
    df_Train_Brand[colYvar] = df_Train_Brand[colYvar].astype(np.uint8)
    df_Train_Brand = df_Train_Brand[['itemid', 'image_path', colYvar]]
    print(df_Train_Brand.dtypes)
    print(f"after {colYvar} dropna", df_Train_Brand.shape)

    # seriesTop3 = df_Train_Brand.groupby('Brand').count()['itemid'].sort_values().tail(3)
    hardcode = [48, 50, 54] # not top 3 but enough data
    seriesTop3 = df_Train_Brand.groupby('Brand').count()['itemid']
    seriesTop3 = seriesTop3[seriesTop3.index.isin(hardcode)].sort_values()
    seriesTop3_80pcnt = (seriesTop3 * 0.8).apply(int)

    dfTrain_Top3_Brand = df_Train_Brand.loc[df_Train_Full['Brand'].isin(seriesTop3.index)]
    print(f"after {colYvar} dropna", dfTrain_Top3_Brand.shape)
    dfTrain_Top3_Brand.reset_index(inplace=True, drop=True)

    dfTrain_Top3_Brand_Train = pd.DataFrame()
    for idx, cnt in seriesTop3_80pcnt.iteritems():
        dfAdd = dfTrain_Top3_Brand.loc[dfTrain_Top3_Brand[colYvar] == idx].head(cnt)
        dfTrain_Top3_Brand_Train = dfTrain_Top3_Brand_Train.append(dfAdd)
        print (dfTrain_Top3_Brand_Train.shape)

    dfTrain_Top3_Brand_Test = dfTrain_Top3_Brand.loc[~dfTrain_Top3_Brand.index.isin(dfTrain_Top3_Brand_Train.index)]
    print("dfTrain_Top3_Brand_Test.shape", dfTrain_Top3_Brand_Test.shape)

    # at this stage we have filtered out the train and test for the top 3 mobile phones brands
    dfTrain_Top3_Brand_Train.loc[:, 'thumbnail'] = dfTrain_Top3_Brand_Train['image_path'].apply(lambda x: Image.open(pathData + x).convert("L"))
    dfTrain_Top3_Brand_Train['thumbnail'].apply(lambda x: x.thumbnail((128, 128), Image.ANTIALIAS))
    dfTrain_Top3_Brand_Train['array'] = dfTrain_Top3_Brand_Train['thumbnail'].apply(np.asarray)
    dfTrain_Top3_Brand_Train['sizes'] = dfTrain_Top3_Brand_Train['array'].apply(lambda x: x.shape)
    del dfTrain_Top3_Brand_Train['thumbnail']

    x_train = np.array(dfTrain_Top3_Brand_Train['array'].tolist())
    print("x_train.shape", x_train.shape, x_train.dtype)

    y_train = np.array(dfTrain_Top3_Brand_Train['Brand'].tolist()).astype(np.uint8)
    print("y_train shape", y_train.shape, y_train.dtype)

    dfTrain_Top3_Brand_Test.loc[:, 'thumbnail'] = dfTrain_Top3_Brand_Test['image_path'].apply(lambda x: Image.open(pathData + x).convert("L"))
    dfTrain_Top3_Brand_Test['thumbnail'].apply(lambda x: x.thumbnail((128, 128), Image.ANTIALIAS))
    dfTrain_Top3_Brand_Test['array'] = dfTrain_Top3_Brand_Test['thumbnail'].apply(np.asarray)
    dfTrain_Top3_Brand_Test['sizes'] = dfTrain_Top3_Brand_Test['array'].apply(lambda x: x.shape)
    del dfTrain_Top3_Brand_Test['thumbnail']

    x_test = np.array(dfTrain_Top3_Brand_Test['array'].tolist())
    print("x_test.shape", x_test.shape, x_test.dtype)

    y_test = np.array(dfTrain_Top3_Brand_Test['Brand'].tolist()).astype(np.uint8)
    print("y_test shape", y_test.shape, y_test.dtype)

    print("Sdfs")

##################################################################################################################
# 80 train 20 test
# brand 43 has 33667. split top 26933 is train bottom is test
# brand 2 has 30398. split top 24318 is train
# brand 55 has 21548 split top 17238 is train
# brand 33 has 14679
# brand 46 has 11982
##################################################################################################################


# print("x_train shape", x_train.shape, x_train.dtype)
# y_train = np.array(df['Brand'].tolist()).astype(np.uint8)
# print("y_train shape", y_train.shape, y_train.dtype)
#
# df = df_Train_Full.tail(100).copy()
# df.loc[:, 'thumbnail'] = df['image_path'].apply(lambda x: Image.open(pathData + x).convert("L"))
# df['thumbnail'].apply(lambda x: x.thumbnail((128, 128), Image.ANTIALIAS))
# df['array'] = df['thumbnail'].apply(np.asarray)
# df['sizes'] = df['array'].apply(lambda x: x.shape)
# del df['thumbnail']
#
# x_test = np.array(df['array'].tolist())
# print("x_test shape", x_test.shape, x_test.dtype)
# y_test = np.array(df['Brand'].tolist()).astype(np.uint8)
# print("y_test shape", y_test.shape, y_test.dtype)


# set some parameters
# Parameters
learning_rate = 0.01
num_steps = 2000
batch_size = 128
display_step = 100

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
num_input = 16384 # MNIST data input (img shape: 128*128)
num_classes = 10 # MNIST total classes (0-9 digits)

