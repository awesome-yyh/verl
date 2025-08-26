'''
Author: yangyahe yangyahe@midu.com
Date: 2025-08-18 17:22:25
LastEditors: yangyahe yangyahe@midu.com
LastEditTime: 2025-08-20 15:49:24
FilePath: /app/yangyahe/verl/scripts/parquet_shuff.py
Description: 随机打乱parquet文件
'''
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np

# merge_path="data/anli_ft_merge_train.parquet"
# merge_path="data/anli_ft_merge_test.parquet"
merge_path="data/anli_ft_correct_train.parquet"
# merge_path="data/anli_ft_correct_test.parquet"
merge_path="data/anli_ft_correct_merge_test_shuffled.parquet"
# merge_path="data/anli_ft_correct_merge_train_shuffled.parquet"

table = pq.read_table(merge_path)

row = table.slice(1, 1).to_pydict()
# 根据需要转换数据格式
print({k: v for k, v in row.items()})

# 生成随机索引并打乱数据
total_rows = len(table)
print(f"ori table len: {total_rows}")
indices = np.random.permutation(total_rows)

# 分批次处理（每批最多100万行）
batch_size = 1_0000
shuffled_batches = []
for i in range(0, total_rows, batch_size):
    batch_indices = indices[i:i + batch_size]
    shuffled_batch = table.take(batch_indices)
    shuffled_batches.append(shuffled_batch)

# 写入新文件
# 合并批次并保存
shuffled_table = pa.concat_tables(shuffled_batches)
save_path = merge_path.replace(".parquet", "_shuffled.parquet")
pq.write_table(shuffled_table, save_path)

row = shuffled_table.slice(1, 1).to_pydict()
# 根据需要转换数据格式
print({k: v for k, v in row.items()})
print(f"shuffled table len: {len(shuffled_table)}")
print(f"已保存到: {save_path}")

# check
import pandas as pd
df2 = pd.read_parquet(save_path)
print(df2.shape)
print(df2.head)
