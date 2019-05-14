# import standard libraries
import os
import sys

# import libraries
import pandas as pd
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from keras.utils import np_utils
from keras.models import Model

import matplotlib.pyplot as plt
print('Tensorflow Version: ', tf.__version__)
tf.logging.set_verbosity(0)
# np.set_printoptions(threshold=np.inf)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)

pathCwd = os.getcwd()
print(pathCwd)


# read train and test dataset for mobile only
df_Train_Full = pd.read_csv(f'{pathCwd}/ndsc-advanced/mobile_data_info_train_competition.csv')
# print(df_Train_Full.head())
# print(df_Train_Full.shape)
# print(df_Train_Full.dtypes)

df_Test_Val = pd.read_csv(f'{pathCwd}/ndsc-advanced/mobile_data_info_val_competition.csv')
# print(df_Test_Val.head())
# print(df_Test_Val.shape)
# print(df_Test_Val.dtypes)

# declare path for images
pathData = 'D:/NDSC_Dataset/'

thumbnailsize = (28, 28)
numcats = 3

# to iterate through later on
listYvarcolumns = ['Operating System', 'Features', 'Network Connections', 'Memory RAM', 'Brand', 'Warranty Period',
                  'Storage Capacity', 'Color Family', 'Phone Model', 'Camera', 'Phone Screen Size']

for colYvar in ['Brand']:  # listYvarcolumns
    # print(colYvar)
    # now just use brand only
    print(f"before {colYvar} dropna", df_Train_Full.shape)
    df_Train_Brand = df_Train_Full.copy()

    hardcode = [19, 40, 3] # not top 3 but enough data
    df_Train_Brand.loc[df_Train_Brand['Brand'].isin([0.0, 1.0, 2.0]), 'Brand'] = np.nan # hardcode

    df_Train_Brand.dropna(subset=[colYvar], inplace=True)
    df_Train_Brand[colYvar] = df_Train_Brand[colYvar].astype(np.uint8)
    df_Train_Brand = df_Train_Brand[['itemid', 'image_path', colYvar]]
    # print(df_Train_Brand.dtypes)
    print(f"after {colYvar} dropna", df_Train_Brand.shape)

    df_Train_Brand.loc[df_Train_Brand['Brand'] == 19, 'Brand'] = 0
    df_Train_Brand.loc[df_Train_Brand['Brand'] == 40, 'Brand'] = 1
    df_Train_Brand.loc[df_Train_Brand['Brand'] == 3, 'Brand'] = 2

    # seriesTop3 = df_Train_Brand.groupby('Brand').count()['itemid'].sort_values().tail(3)

    seriesTop3 = df_Train_Brand.groupby('Brand').count()['itemid'].sort_values()
    seriesTop3 = seriesTop3[seriesTop3.index.isin([0,1,2])].sort_values()
    seriesTop3_80pcnt = (seriesTop3 * 0.8).apply(int)

    dfTrain_Top3_Brand = df_Train_Brand.loc[df_Train_Brand['Brand'].isin(seriesTop3.index)]
    print(f"after {colYvar} dropna", dfTrain_Top3_Brand.shape)
    dfTrain_Top3_Brand.reset_index(inplace=True, drop=True)

    dfTrain_Top3_Brand_Train = pd.DataFrame()
    for idx, cnt in seriesTop3_80pcnt.iteritems():
        dfAdd = dfTrain_Top3_Brand.loc[dfTrain_Top3_Brand[colYvar] == idx].head(cnt)
        dfTrain_Top3_Brand_Train = dfTrain_Top3_Brand_Train.append(dfAdd)
        print (dfTrain_Top3_Brand_Train.shape)

    dfTrain_Top3_Brand_Test = dfTrain_Top3_Brand.loc[~dfTrain_Top3_Brand.index.isin(dfTrain_Top3_Brand_Train.index)].copy()
    print("dfTrain_Top3_Brand_Test.shape", dfTrain_Top3_Brand_Test.shape)

    # at this stage we have filtered out the train and test for the top 3 mobile phones brands
    dfTrain_Top3_Brand_Train.loc[:, 'thumbnail'] = dfTrain_Top3_Brand_Train['image_path'].apply(lambda x: Image.open(pathData + x).convert("L"))
    dfTrain_Top3_Brand_Train['thumbnail'].apply(lambda x: x.thumbnail(thumbnailsize, Image.ANTIALIAS))
    dfTrain_Top3_Brand_Train['array'] = dfTrain_Top3_Brand_Train['thumbnail'].apply(np.asarray)
    dfTrain_Top3_Brand_Train['sizes'] = dfTrain_Top3_Brand_Train['array'].apply(lambda x: x.shape)
    del dfTrain_Top3_Brand_Train['thumbnail']

    uniqSizes = dfTrain_Top3_Brand_Train['sizes'].unique()
    if len(uniqSizes) < 1:
        print("uniqSizes dfTrain_Top3_Brand_Train problem", uniqSizes)
        # dfTrain_Top3_Brand_Train
        exit()

    x_train = np.array(dfTrain_Top3_Brand_Train['array'].tolist())
    print("x_train.shape", x_train.shape, x_train.dtype)

    y_train = np.array(dfTrain_Top3_Brand_Train['Brand'].tolist())
    print("y_train shape", y_train.shape, y_train.dtype)

    dfTrain_Top3_Brand_Test.loc[:, 'thumbnail'] = dfTrain_Top3_Brand_Test['image_path'].apply(lambda x: Image.open(pathData + x).convert("L"))
    dfTrain_Top3_Brand_Test['thumbnail'].apply(lambda x: x.thumbnail(thumbnailsize, Image.ANTIALIAS))
    dfTrain_Top3_Brand_Test['array'] = dfTrain_Top3_Brand_Test['thumbnail'].apply(np.asarray)
    dfTrain_Top3_Brand_Test['sizes'] = dfTrain_Top3_Brand_Test['array'].apply(lambda x: x.shape)
    del dfTrain_Top3_Brand_Test['thumbnail']

    uniqSizes = dfTrain_Top3_Brand_Test['sizes'].unique()
    if len(uniqSizes) < 1:
        print("uniqSizes dfTrain_Top3_Brand_Test problem", uniqSizes)
        exit()

    x_test = np.array(dfTrain_Top3_Brand_Test['array'].tolist())
    print("x_test.shape", x_test.shape, x_test.dtype)

    y_test = np.array(dfTrain_Top3_Brand_Test['Brand'].tolist()).astype(np.uint8)
    print("y_test shape", y_test.shape, y_test.dtype)

    # Do some simple Preprocessing
    # Normalize the data
    train_images = x_train / 255.0
    test_images = x_test / 255.0

    # convert labels to one-hot
    train_labels = np_utils.to_categorical(y_train)
    test_labels = np_utils.to_categorical(y_test)

    print('Shape of train_images:', train_images.shape)
    print('Shape of train_labels:', train_labels.shape)
    print('Shape of test_images:', test_images.shape)
    print('Shape of test_labels:', test_labels.shape)

    # build model
    # ! Important: must define the input shape in the first layer
    model_MLP = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28,)),
        keras.layers.Dense(128, input_shape=(train_images.shape[1],), activation='relu'),
        keras.layers.Dense(numcats, activation='softmax')]
    )
    model_MLP.add(keras.layers.Dense(128, input_shape=(train_images.shape[1],), activation='relu'))
    model_MLP.add(keras.layers.Dense(numcats, activation='softmax'))
    model_MLP.compile(optimizer='adam', loss='categorical_crossentropy',
                      metrics=['accuracy'])
    print(model_MLP.summary())

    # train the model
    model_MLP.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels), batch_size=32)

    # reshpe
    train_images = np.reshape(train_images, (train_images.shape[0], 28, 28, 1))
    test_images = np.reshape(test_images, (test_images.shape[0], 28, 28, 1))

    # Build a CNN model
    # first cnn layer
    model_cnn = tf.keras.Sequential()
    model_cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2,
                                         padding='same', activation='relu',
                                         input_shape=(28, 28, 1)))
    model_cnn.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model_cnn.add(tf.keras.layers.Dropout(0.3))

    # second cnn layer
    model_cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2,
                                         padding='same', activation='relu'))
    model_cnn.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model_cnn.add(tf.keras.layers.Dropout(0.3))

    # add FC layer
    model_cnn.add(tf.keras.layers.Flatten())
    model_cnn.add(tf.keras.layers.Dense(256, activation='relu'))
    model_cnn.add(tf.keras.layers.Dropout(0.5))

    # output layer
    model_cnn.add(tf.keras.layers.Dense(numcats, activation='softmax'))
    model_cnn.summary()

    model_cnn.compile(loss='categorical_crossentropy', optimizer='adam',
                      metrics=['accuracy'])

    model_cnn.fit(train_images, train_labels, batch_size=64, epochs=10,
                  validation_data=(test_images, test_labels))

    # Reimplement it using funcitonal api

    # Input layer
    inputs = tf.keras.layers.Input(shape=(28, 28, 1,))
    # first convolution layer
    conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=2,
                                   padding='same', activation='relu')(inputs)
    max_pool1 = tf.keras.layers.MaxPooling2D(pool_size=2)(conv1)
    dropout1 = tf.keras.layers.Dropout(0.3)(max_pool1)
    conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=2,
                                   padding='same', activation='relu')(dropout1)
    max_pool2 = tf.keras.layers.MaxPooling2D(pool_size=2)(conv2)
    dropout2 = tf.keras.layers.Dropout(0.3)(max_pool2)
    flat1 = tf.keras.layers.Flatten()(dropout2)
    fc1 = tf.keras.layers.Dense(256, activation='relu')(flat1)
    dropout3 = tf.keras.layers.Dropout(0.5)(fc1)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(dropout3)

    # build model and compile
    model_cnn_fun = Model(inputs=inputs, outputs=outputs)
    model_cnn_fun.compile(loss='categorical_crossentropy', optimizer='adam',
                          metrics=['accuracy'])
    model_cnn_fun.summary()

    model_cnn_fun.fit(train_images, train_labels, batch_size=64, epochs=10,
                      validation_data=(test_images, test_labels))