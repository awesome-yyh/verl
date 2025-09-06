import pandas as pd

# 读取 Parquet 文件
merge_path = "data/anli_ft_lanjie_merge_x_train_deduplicated_shuffled_all2x_deduplicated_shuffled_error_correct_noop.parquet"

df = pd.read_parquet(merge_path)

# 获取前 100 行（如果行数不足则取全部）
result = df.head(100)

# 保存到新的 Parquet 文件
output_path = merge_path.replace('train', 'test')
result.to_parquet(output_path, index=False)
print(f"已保存前 100 行到 {output_path}")