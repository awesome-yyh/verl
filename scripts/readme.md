jsonl2csv
term-corrector/scripts/wenxiux/jsonl2csv.py

file = "data/案例收集/误漏报反馈/merge_wubao.csv_replace_key.txt"  # 4255 行 魔方勘误表误报数据，只有一列正确句
file = "data/案例收集/误漏报反馈/merge_loubao.csv_replace_key.txt"  # 322 行 魔方勘误表漏报数据（其他模型正报），2列（错误句\t正确句）

1. 通过插件可以获取每个原句的各个模型输出句（得到csv文件）

2. 获取大模型的解释句（得到jsonl文件）
term-corrector/scripts/wenxiux/create_data_from_llm.py

3. 转换为parquet文件（得到最终训练的parquet文件）
verl/scripts/txt2parquet_merge.py
verl/scripts/txt2parquet_correct.py

4. 合并、去重、打散、检查
verl/scripts/parquet_merge.py
verl/scripts/parquet_dedup.py
verl/scripts/parquet_shuff.py
verl/scripts/parquet_check.py
