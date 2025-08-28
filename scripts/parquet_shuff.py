'''
Author: yangyahe yangyahe@midu.com
Date: 2025-08-18 17:22:25
LastEditors: yangyahe yangyahe@midu.com
LastEditTime: 2025-08-20 15:49:24
FilePath: /app/yangyahe/verl/scripts/parquet_shuff.py
Description: 随机打乱parquet文件
'''

import pandas as pd
import numpy as np

# 文件路径
# merge_path="data/anli_ft_merge_train.parquet"
# merge_path="data/anli_ft_merge_test.parquet"
merge_path="data/anli_ft_correct_train.parquet"
# merge_path="data/anli_ft_correct_test.parquet"
merge_path="data/anli_ft_correct_merge_test_shuffled.parquet"
# merge_path="data/anli_ft_correct_merge_train_shuffled.parquet"
merge_path="data/www_merge_拦截_correct_merge_train_deduplicated.parquet"

save_path = merge_path.replace(".parquet", "_shuffled.parquet")

# 使用 Pandas 读取 Parquet 文件
print("读取数据...")
df = pd.read_parquet(merge_path)

# 显示原始数据信息
print(f"原始数据形状: {df.shape}")
print("前几行数据:")
print(df.head(2))
print("\n列信息:")
print(df.dtypes)

# 使用 Pandas 的 sample 方法直接打散数据（更高效）
print("\n打散数据...")
shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 保存打散后的数据
print("保存数据...")
shuffled_df.to_parquet(save_path, index=False)

# 验证结果
print(f"已保存到: {save_path}")
print(f"打散后数据形状: {shuffled_df.shape}")
print("打散后的前几行数据:")
print(shuffled_df.head(10))

# 可选：验证数据完整性
print("\n验证数据完整性...")
original_count = len(df)
shuffled_count = len(shuffled_df)
print(f"原始数据行数: {original_count}")
print(f"打散后数据行数: {shuffled_count}")
print(f"数据完整性: {'通过' if original_count == shuffled_count else '失败'}")