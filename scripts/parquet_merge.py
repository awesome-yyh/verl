'''
Author: yangyahe yangyahe@midu.com
Date: 2025-08-20 15:20:09
LastEditors: yangyahe yangyahe@midu.com
LastEditTime: 2025-08-20 15:41:30
FilePath: /app/yangyahe/verl/scripts/parquet_merge.py
Description: 
合并纠错和魔方数据，
注意：还需要shuff
'''
import pandas as pd

# 读取两个Parquet文件
df1 = pd.read_parquet('data/anli_ft_correct_train_shuffled.parquet')
print(df1.shape)
print(df1.head)

df2 = pd.read_parquet('data/anli_ft_merge_train_shuffled.parquet')
print(df2.shape)
print(df2.head)

# 合并两个DataFrame
df_combined = pd.concat((df1, df2), ignore_index=True)

# 将合并后的数据写入新的Parquet文件
df_combined.to_parquet('data/anli_ft_correct_merge_train_shuffled.parquet')
print(df_combined.shape)
print(df_combined.head)
