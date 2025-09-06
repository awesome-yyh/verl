"""
修改整个parquet文件的ability
"""

merge_path="data/anli_ft_correct_merge_train_shuffled_shuffled.parquet"  # 6w0156, x, 待去重，待修改，ability，src == tgt 的比例: 41.83%，ability='correct_merge'
merge_path="data/anli_ft_correct_train.parquet"  # 12w2078, x, 待去重，，待修改，ability，src == tgt 的比例: 52.11% ability: correct_merge
merge_path = "data/anli_ft_lanjie_merge_x_train_deduplicated_shuffled.parquet"
output_path = "data/anli_ft_lanjie_merge_x_train_deduplicated_shuffled_all2x.parquet"
# 这个parquet文件有一列字段ability，将其内容都修改为'correct_x', 其他字段不变

import pandas as pd
df = pd.read_parquet(merge_path)
df['ability'] = 'correct_x'
df.to_parquet(output_path, index=False)
print(f"去重后的文件已保存至: {output_path}")

print("\n第30行详细信息：")
third_row = df.iloc[29]
for column, value in third_row.items():
    print(f"{column}: {value} \n")
