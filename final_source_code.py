import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import json
import pprint
import math
import os
import time
# print(os.listdir('/new_software/Shopee Competition'))
pd.options.mode.chained_assignment = None

tnow = time.time()

def generateDf(filename, series_list):
    drive_path = filename
    # category = filename.split('_')[0]
    #     For testing of 100 rows from each file
    df_category = pd.read_csv(drive_path)
    # Adding on columns with default values as nan
    for series in series_list:
        df_category[series] = np.nan
    return df_category


def generateSubmissionFormatFromDf(dfList, categories):
    df_concat = pd.DataFrame(columns=['id', 'tagging'])
    category_count = 0
    for df_category in dfList:
        df_cols = df_category.columns.values
        # print(df_category.count().sum())
        df_temp = pd.DataFrame()
        for i in range(len(df_category.columns)):
            if i > 2:
                # print('_' + df_cols[i])
                if categories[category_count] == 'beauty':
                    df_temp['id'] = [str(s) + '_' + df_cols[i].replace(" ", "_") for s in df_category.itemid]
                else:
                    df_temp['id'] = [str(s) + '_' + df_cols[i] for s in df_category.itemid]
                df_temp['tagging'] = [predict for predict in df_category.iloc[:, i]]
                df_concat = pd.concat([df_concat, df_temp])
        category_count = category_count + 1
    df_concat = df_concat.sort_values('id').reset_index(drop=True)
    bool_series = pd.isnull(df_concat['tagging'])
    df_concat['tagging'][bool_series] = 0
    df_concat['tagging'] = df_concat['tagging'].astype(int)
    return df_concat


# def generateSubmissionFormat(listOfData, path='/new_software/Shopee Competition/'):
#     df_concat = pd.DataFrame(columns=['id', 'tagging'])
#     for filename in listOfData:
#         drive_path = path + filename
#         category = filename.split('_')[0]
#         #     For testing of 100 rows from each file
#         df_category = pd.read_csv(drive_path, nrows=100)
#         # For loading the full data from CSV files
#         #     df_category = pd.read_csv(drive_path)
#         df_cols = df_category.columns.values
#
#         # print(df_category.count().sum())
#         df_temp = pd.DataFrame()
#         for i in range(len(df_category.columns)):
#             if i > 2:
#                 # print('_' + df_cols[i])
#                 if category == 'beauty':
#                     df_temp['id'] = [str(s) + '_' + df_cols[i].replace(" ", "_") for s in df_category.itemid]
#                 else:
#                     df_temp['id'] = [str(s) + '_' + df_cols[i] for s in df_category.itemid]
#                 df_temp['tagging'] = [predict for predict in df_category.iloc[:, i]]
#                 # print(df_temp.head())
#                 df_concat = pd.concat([df_concat, df_temp])
#     df_concat = df_concat.sort_values('id').reset_index(drop=True)
#     bool_series = pd.isnull(df_concat['tagging'])
#     df_concat['tagging'][bool_series] = 0
#     df_concat['tagging'] = df_concat['tagging'].astype(int)
#     return df_concat

def df_word_process(df, dfjson):
    for index, row in df.iterrows():
        title = row["title"].lower()
        # print(title)
        for first_word in dfjson['first_word'].unique():
            if first_word in title:
                try:
                    # print("first_word found ",first_word)
                    another_list = dfjson[dfjson["first_word"]== first_word].index
                    for attribute in another_list:
                        if attribute in title:
                            dfjsonT = dfjson.T.copy()
                            value = dfjsonT.loc[lambda dfjsonT: dfjsonT[attribute].notnull(),attribute].values[0]
                            col_attribute = dfjson[dfjson.index==attribute]["type"][0]
                            df[col_attribute][index] = value
                            # print("\t attribute found ",col_attribute, " with value ",value )
                            break
                        else:
                            # print("first word found ",first_word, " but no specific attribute ",attribute," in ",title)
                            pass
                except:
                    # print("error detected")
                    pass
            else:
                # print("first_word not found ",first_word )
                pass
    return df

def get_json_from_filename(filename_json):
    drive_path_json = filename_json
    dfjson = pd.read_json(drive_path_json, orient="columns")
    # dfjsonT = dfjson.T
    dfjson['first_word'] = dfjson.index.str.split().str.get(0)
    dfjson["type"] = dfjson.apply(lambda row: row.first_valid_index(), axis=1)
    return dfjson

dfjson_mobile = get_json_from_filename('ndsc-advanced/mobile_profile_train.json')
dfjson_fashion = get_json_from_filename('ndsc-advanced/fashion_profile_train.json')
dfjson_beauty = get_json_from_filename('ndsc-advanced/beauty_profile_train.json')

# Add on specifications
mobile_specifications = ['Operating System', 'Features', 'Network Connections', 'Memory RAM', 'Warranty Period',
                         'Storage Capacity', 'Color Family', 'Camera', 'Phone Screen Size']
fashion_specifications = ['Pattern', 'Collar Type', 'Fashion Trend', 'Clothing Material','Sleeves']
beauty_specifications = ['Benefits', 'Brand', 'Colour_group', 'Product_texture', 'Skin_type']

# Read the df category mobile
df_category_val_mobile = generateDf('ndsc-advanced/mobile_data_info_val_competition.csv',mobile_specifications)
df_category_val_mobile_100 = df_category_val_mobile.head(1000)
df_category_val_mobile_10 = df_category_val_mobile.head(10)
df_category_val_mobile_5 = df_category_val_mobile.head(5)

# Read the df category beauty
df_category_val_beauty = generateDf('ndsc-advanced/beauty_data_info_val_competition.csv',beauty_specifications)
df_category_val_beauty_100 = df_category_val_beauty.head(1000)
df_category_val_beauty_10 = df_category_val_beauty.head(10)
df_category_val_beauty_5 = df_category_val_beauty.head(5)

# Read the df category fashine
df_category_val_fashion = generateDf('ndsc-advanced/fashion_data_info_val_competition.csv',fashion_specifications)
df_category_val_fashion_100 = df_category_val_fashion.head(1000)
df_category_val_fashion_10 = df_category_val_fashion.head(10)
df_category_val_fashion_5 = df_category_val_fashion.head(5)

#######################################################################################################################
# Word processing for each mobile, beauty and fashion
df_category_val_mobile = df_word_process(df_category_val_mobile, dfjson_mobile)
print(f"mobile took {time.time() - tnow} seconds")

# Listings df to be transformed to submission formats with categories label
list_df_submission = [df_category_val_mobile]
categories = ['mobile']
df_submission_mobile = generateSubmissionFormatFromDf(list_df_submission, categories)
# Print submission to csv
print (df_submission_mobile.shape)
df_submission_mobile.to_csv('submission_mobile.csv', index=False)
print("done!")
print(f"mobile to df took {time.time() - tnow} seconds")

#######################################################################################################################

df_category_val_beauty = df_word_process(df_category_val_beauty, dfjson_beauty)
print(f"beauty took {time.time() - tnow} seconds")

# Listings df to be transformed to submission formats with categories label
list_df_submission = [df_category_val_beauty]
categories = ['beauty']
df_submission_beauty = generateSubmissionFormatFromDf(list_df_submission, categories)
# Print submission to csv
print (df_submission_beauty.shape)
df_submission_beauty.to_csv('submission_beauty.csv', index=False)
print("done!")
print(f"beauty to df took {time.time() - tnow} seconds")

#######################################################################################################################

df_category_val_fashion = df_word_process(df_category_val_fashion, dfjson_fashion)
print(f"fashion took {time.time() - tnow} seconds")

# Listings df to be transformed to submission formats with categories label
list_df_submission = [df_category_val_fashion]
categories = ['fashion']
df_submission_fashion = generateSubmissionFormatFromDf(list_df_submission, categories)
# Print submission to csv
print (df_submission_fashion.shape)
df_submission_fashion.to_csv('submission_fashion.csv', index=False)
print("done!")
print(f"fashion to df took {time.time() - tnow} seconds")

#######################################################################################################################

dfSUBMIT = pd.DataFrame()
dfSUBMIT = dfSUBMIT.append(df_submission_mobile)
dfSUBMIT = dfSUBMIT.append(df_submission_beauty)
dfSUBMIT = dfSUBMIT.append(df_submission_fashion)
dfSUBMIT.to_csv('submission_FINAL.csv', index=False)
print (dfSUBMIT.shape)
print("done!")
print(f"FINAL to df took {time.time() - tnow} seconds")
# # Listings df to be transformed to submission formats with categories label
# list_df_submission = [df_category_val_mobile, df_category_val_beauty, df_category_val_fashion]
# categories = ['mobile', 'beauty', 'fashion']
# df_submission = generateSubmissionFormatFromDf(list_df_submission, categories)
#
# # Print submission to csv
# print (df_submission.shape)
# df_submission.to_csv('submission.csv')
# print("done!")
# print(f"took {time.time() - tnow} seconds")

# df_submission = generateSubmissionFormat(['mobile_data_info_train_competition.csv','fashion_data_info_train_competition.csv','beauty_data_info_train_competition.csv'])
# df_submission = generateSubmissionFormat(['mobile_data_info_train_competition.csv','beauty_data_info_train_competition.csv'])

