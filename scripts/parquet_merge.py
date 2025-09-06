import pandas as pd
from pathlib import Path

def merge_parquet_files(input_files, output_file):
    """
    合并多个Parquet文件为一个
    
    参数:
        input_files: 包含Parquet文件路径的列表
        output_file: 合并后的输出文件路径
    """
    # 存储所有DataFrame的列表
    dfs = []
    
    # 遍历并读取每个Parquet文件
    for file_path in input_files:
        # 检查文件是否存在
        if not Path(file_path).exists():
            print(f"警告: 文件 {file_path} 不存在，已跳过")
            continue
            
        # 读取Parquet文件
        df = pd.read_parquet(file_path)
        dfs.append(df)
        
        # 打印当前文件信息
        print(f"已读取 {file_path}，形状: {df.shape}")
        print(df.head)
    
    if not dfs:
        print("错误: 没有有效的文件可合并")
        return
    
    # 合并所有DataFrame
    df_combined = pd.concat(dfs, ignore_index=True)
    
    # 确保输出目录存在
    output_dir = Path(output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存合并后的文件
    df_combined.to_parquet(output_file)
    
    # 打印合并结果信息
    print(f"\n合并完成，总形状: {df_combined.shape}")
    print(f"合并后文件已保存至: {output_file}")
    print("前5行数据:")
    print(df_combined.head())

if __name__ == "__main__":
    # 在这里添加所有需要合并的Parquet文件路径
    parquet_files = [
        # 'data/anli_ft_correct_train_shuffled.parquet',
        # 'data/anli_ft_merge_train_shuffled.parquet',
        'data/www_merge正报案例_6c95cea9-572b-4d52-8b06-40662069d770.csv_test_doubao_explanation-x_train.parquet',
        'data/魔方拦截案例_f97589df-e3b8-47e2-aba4-62197fa5d667.csv_test_doubao_explanation-x_train.parquet',
        'data/魔方误拦截all.csv_015f309d-b27d-4f16-9381-c3a737d403d8.txt_test_doubao_explanation-x_train.parquet',
        'data/merge_wubao.csv_replace_key_acd4168d-f405-4791-afce-6f2c8e59fbad.csv_test_doubao_explanation-x_train.parquet',
        'data/merge_loubao.csv_replace_key_0df02291-7803-4eec-94c1-923c13226849.csv_test_doubao_explanation-x_train.parquet',
        'data/www_merge正报案例_6c95cea9-572b-4d52-8b06-40662069d770.csv_test_doubao_explanation_train.parquet',
        'data/魔方拦截案例_f97589df-e3b8-47e2-aba4-62197fa5d667.csv_test_doubao_explanation-merge_train.parquet',
        'data/魔方误拦截all.csv_015f309d-b27d-4f16-9381-c3a737d403d8.txt_test_doubao_explanation-merge_train.parquet',
        'data/merge_wubao.csv_replace_key_acd4168d-f405-4791-afce-6f2c8e59fbad.csv_test_doubao_explanation-merge_train.parquet',
        'data/merge_loubao.csv_replace_key_0df02291-7803-4eec-94c1-923c13226849.csv_test_doubao_explanation-merge_train.parquet',
    ]
    
    # 合并后的输出文件路径
    output_file = 'data/www_merge_拦截_correct_merge_train.parquet'
    
    # 执行合并
    merge_parquet_files(parquet_files, output_file)

    print(f"合并完成: {output_file}")