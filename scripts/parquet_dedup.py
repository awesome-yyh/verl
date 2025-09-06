import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm

def deduplicate_parquet(input_path, output_path, keep='first'):
    """
    对Parquet文件进行去重，基于"extra_info.src"和"ability"字段复合内容唯一
    
    参数:
        input_path: 输入Parquet文件路径
        output_path: 去重后输出文件路径
        keep: 保留策略，'first'保留第一个出现的记录，'last'保留最后一个
    """
    # 检查输入文件是否存在
    if not Path(input_path).exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    
    # 读取Parquet文件
    print(f"正在读取文件: {input_path}")
    df = pd.read_parquet(input_path)
    
    # 检查是否存在必要的字段
    if "extra_info" not in df.columns:
        raise ValueError("数据中不包含'extra_info'字段")
    if "ability" not in df.columns:
        raise ValueError("数据中不包含'ability'字段")
    
    # 提取extra_info中的src字段并添加为新列，用于去重
    print("正在提取extra_info.src字段...")
    tqdm.pandas(desc="处理记录")
    
    try:
        # 处理extra_info字段，提取src值
        # 考虑到可能存储为字符串或字典类型，做双重处理
        def extract_src(extra_info):
            if pd.isna(extra_info):
                return None
            if isinstance(extra_info, str):
                # 如果是字符串形式的JSON，先解析
                try:
                    info_dict = json.loads(extra_info)
                    return info_dict.get("src", None)
                except (json.JSONDecodeError, TypeError):
                    return None
            elif isinstance(extra_info, dict):
                # 如果是字典类型，直接获取
                return extra_info.get("src", None)
            return None
        
        # 应用提取函数
        df["_src_to_dedup"] = df["extra_info"].progress_apply(extract_src)
        
        # 统计去重前的记录数
        original_count = len(df)
        print(f"去重前记录数: {original_count}")
        
        # 检查是否有无效的src值
        invalid_src_count = df["_src_to_dedup"].isna().sum()
        if invalid_src_count > 0:
            print(f"警告: 发现 {invalid_src_count} 条记录的extra_info.src字段无效或缺失")
        
        # 检查是否有无效的ability值
        invalid_ability_count = df["ability"].isna().sum()
        if invalid_ability_count > 0:
            print(f"警告: 发现 {invalid_ability_count} 条记录的ability字段无效或缺失")
        
        # 创建复合键用于去重
        print("创建复合键用于去重...")
        df["_composite_key"] = df["_src_to_dedup"].astype(str) + "|" + df["ability"].astype(str)
        
        # 执行去重（基于复合键）
        df_deduplicated = df.drop_duplicates(
            subset=["_composite_key"], 
            keep=keep,
            ignore_index=True
        )
        
        # 移除临时列
        df_deduplicated = df_deduplicated.drop(columns=["_src_to_dedup", "_composite_key"])
        
        # 统计去重后的记录数
        dedup_count = len(df_deduplicated)
        removed_count = original_count - dedup_count
        print(f"去重后记录数: {dedup_count}")
        print(f"已移除重复记录数: {removed_count}")
        print(f"重复率: {removed_count/original_count:.2%}")
        
        # 确保输出目录存在
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存去重后的文件
        df_deduplicated.to_parquet(output_path, index=False)
        print(f"去重后的文件已保存至: {output_path}")
        
        return df_deduplicated
        
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        raise

if __name__ == "__main__":
    # 配置文件路径
    input_file = "data/www_merge_拦截_correct_merge_train.parquet"
    input_file = "data/www_merge_拦截_correct_merge_train_anli_ft.parquet"
    input_file = "data/anli_ft_lanjie_merge_x_train.parquet"
    input_file = "data/anli_ft_lanjie_merge_x_train_deduplicated_shuffled_all2x.parquet"
    output_file = input_file.replace(".parquet", "_deduplicated.parquet")  # 去重后的输出文件路径
    
    # 执行去重
    deduplicate_parquet(
        input_path=input_file,
        output_path=output_file,
        keep='first'  # 保留第一个出现的记录
    )
