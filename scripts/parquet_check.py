import pandas as pd
import json
from collections import Counter

# 文件路径
# merge_path="data/anli_ft_correct_merge_train_shuffled_shuffled.parquet"
# merge_path="data/anli_ft_correct_merge_test_shuffled_shuffled.parquet"
merge_path="data/www_merge_拦截_correct_merge_train_deduplicated_shuffled.parquet"
merge_path="data/www_merge_拦截_correct_merge_train_anli_ft_deduplicated_shuffled.parquet"

print("读取数据...")
df = pd.read_parquet(merge_path)
print(df.head(2))
print(f"数据形状: {df.shape}")
print("\n列名:")
print(df.columns.tolist())

print("\n第3行详细信息：")
third_row = df.iloc[2]
for column, value in third_row.items():
    print(f"{column}: {value} \n")

# 检查重复率、两个任务(ability)各自的数量和比例、对于merge 包含正确句的比例、正确到正确的比例；对于x 正确到正确的比例；

# 检查 extra_info.src 和 ability 复合字段 的重复率
if 'extra_info' in df.columns and 'ability' in df.columns:
    print("\n=== 分析 extra_info.src 和 ability 复合字段的重复率 ===")
    
    # 提取复合键 (src + ability)
    composite_keys = []
    invalid_count = 0
    
    for idx, (info, ability) in enumerate(zip(df['extra_info'], df['ability'])):
        try:
            # 处理 extra_info 字段，提取 src
            if pd.isna(info):
                src_value = None
            elif isinstance(info, str):
                try:
                    info_dict = json.loads(info)
                    src_value = info_dict.get("src", None)
                except (json.JSONDecodeError, TypeError):
                    src_value = None
            elif isinstance(info, dict):
                src_value = info.get("src", None)
            else:
                src_value = None
            
            # 处理 ability 字段
            ability_value = ability if not pd.isna(ability) else None
            
            # 创建复合键
            if src_value is not None and ability_value is not None:
                composite_key = f"{src_value}|||{ability_value}"
            else:
                composite_key = None
                invalid_count += 1
                
            composite_keys.append(composite_key)
            
        except Exception as e:
            print(f"行 {idx}: 处理失败 - {str(e)}")
            composite_keys.append(None)
            invalid_count += 1
    
    # 计算复合键的重复率
    composite_series = pd.Series(composite_keys)
    total_count = len(composite_series)
    valid_count = total_count - composite_series.isna().sum()
    unique_count = composite_series.nunique()
    duplicate_count = valid_count - unique_count
    duplicate_rate = duplicate_count / valid_count if valid_count > 0 else 0
    
    print(f"总记录数: {total_count}")
    print(f"有效复合键数量: {valid_count}")
    print(f"无效复合键数量: {invalid_count}")
    print(f"唯一复合键数量: {unique_count}")
    print(f"重复复合键数量: {duplicate_count}")
    print(f"复合键重复率: {duplicate_rate:.4f} ({duplicate_rate*100:.2f}%)")
    
    # 显示最常见的复合键
    print("\n最常见的 10 个复合键:")
    composite_counts = composite_series.value_counts()
    for composite, count in composite_counts.head(10).items():
        if pd.isna(composite):
            continue
        src, ability = composite.split("|||")
        print(f"  src: '{src}', ability: '{ability}' - 出现 {count} 次")
    
    # 显示重复的复合键
    duplicates = composite_counts[composite_counts > 1]
    duplicates = duplicates[~duplicates.index.isna()]  # 排除空值
    
    if not duplicates.empty:
        print(f"\n有 {len(duplicates)} 个重复的复合键")
        print("重复的复合键示例:")
        for composite, count in duplicates.head(5).items():
            src, ability = composite.split("|||")
            print(f"  src: '{src}', ability: '{ability}' - 重复 {count} 次")
    else:
        print("\n没有重复的复合键")
        
    # 额外分析：按 ability 分组统计重复率
    print("\n=== 按 ability 分组的重复率分析 ===")
    ability_groups = {}
    for composite_key in composite_keys:
        if composite_key is None:
            continue
        src, ability = composite_key.split("|||")
        if ability not in ability_groups:
            ability_groups[ability] = []
        ability_groups[ability].append(composite_key)
    
    for ability, keys in ability_groups.items():
        total = len(keys)
        unique = len(set(keys))
        dup_rate = (total - unique) / total if total > 0 else 0
        print(f"ability '{ability}': {total} 条记录, {unique} 个唯一复合键, "
              f"重复率 {dup_rate:.4f} ({dup_rate*100:.2f}%)")
elif 'extra_info' not in df.columns:
    print("数据中没有 extra_info 字段")
elif 'ability' not in df.columns:
    print("数据中没有 ability 字段")

# 检查 ability 字段的种类及数量
if 'ability' in df.columns:
    print("\n=== 分析 ability 字段 ===")
    
    ability_counts = df['ability'].value_counts()
    total_abilities = len(ability_counts)
    
    print(f"ability 种类数量: {total_abilities}")
    print("\n各种 ability 的数量:")
    for ability, count in ability_counts.items():
        print(f"  {ability}: {count}")
    
    # 计算占比
    print("\n各种 ability 的占比:")
    for ability, count in ability_counts.items():
        percentage = count / len(df) * 100
        print(f"  {ability}: {percentage:.2f}%")
else:
    print("数据中没有 ability 字段")

def analyze_ability_specific_stats(df):
    """
    分析特定 ability 字段的 extra_info 属性比例
    
    参数:
        df: 包含数据的 DataFrame
    """
    results = {}
    
    # 检查数据中是否包含必要的字段
    if 'extra_info' not in df.columns or 'ability' not in df.columns:
        print("数据中缺少必要的字段: 'extra_info' 或 'ability'")
        return results
    
    # 处理 correct_x ability
    if 'correct_x' in df['ability'].values:
        print("\n=== 分析 ability='correct_x' 的记录 ===")
        correct_x_df = df[df['ability'] == 'correct_x']
        print(f"找到 {len(correct_x_df)} 条 ability='correct_x' 的记录")
        
        # 检查 src == tgt 的比例
        src_tgt_match_count = 0
        total_count = 0
        
        for idx, row in correct_x_df.iterrows():
            try:
                extra_info = row['extra_info']
                
                # 处理 extra_info 字段
                if isinstance(extra_info, str):
                    info_dict = json.loads(extra_info)
                else:
                    info_dict = extra_info
                
                if 'src' in info_dict and 'tgt' in info_dict:
                    total_count += 1
                    if info_dict['src'] == info_dict['tgt']:
                        src_tgt_match_count += 1
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                print(f"行 {idx}: 解析 extra_info 失败 - {str(e)}")
                continue
        
        if total_count > 0:
            match_ratio = src_tgt_match_count / total_count
            print(f"src == tgt 的比例: {match_ratio:.4f} ({match_ratio*100:.2f}%)")
            results['correct_x_src_tgt_match_ratio'] = match_ratio
        else:
            print("没有找到有效的 src 和 tgt 字段")
    
    # 处理 correct_merge ability
    if 'correct_merge' in df['ability'].values:
        print("\n=== 分析 ability='correct_merge' 的记录 ===")
        correct_merge_df = df[df['ability'] == 'correct_merge']
        print(f"找到 {len(correct_merge_df)} 条 ability='correct_merge' 的记录")
        
        # 检查 src == tgt 的比例
        src_tgt_match_count = 0
        total_count = 0
        
        # 检查 jdt_map values 包含 tgt 的比例
        jdt_contains_tgt_count = 0
        jdt_total_count = 0
        
        for idx, row in correct_merge_df.iterrows():
            try:
                extra_info = row['extra_info']
                
                # 处理 extra_info 字段
                if isinstance(extra_info, str):
                    info_dict = json.loads(extra_info)
                else:
                    info_dict = extra_info
                
                # 检查 src == tgt
                if 'src' in info_dict and 'tgt' in info_dict:
                    total_count += 1
                    if info_dict['src'] == info_dict['tgt']:
                        src_tgt_match_count += 1
                
                # 检查 jdt_map values 包含 tgt
                if 'source_line' in info_dict:
                    try:
                        source_line = json.loads(info_dict['source_line'])
                        if 'jdt_map' in source_line:
                            jdt_map = source_line['jdt_map']
                            jdt_total_count += 1
                            
                            # 检查 jdt_map 的 values 是否包含 tgt
                            if 'tgt' in info_dict and info_dict['tgt'] in jdt_map.values():
                                jdt_contains_tgt_count += 1
                    except (json.JSONDecodeError, TypeError, ValueError):
                        # source_line 解析失败，跳过
                        pass
                        
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                print(f"行 {idx}: 解析 extra_info 失败 - {str(e)}")
                continue
        
        # 输出 src == tgt 的结果
        if total_count > 0:
            match_ratio = src_tgt_match_count / total_count
            print(f"src == tgt 的比例: {match_ratio:.4f} ({match_ratio*100:.2f}%)")
            results['correct_merge_src_tgt_match_ratio'] = match_ratio
        else:
            print("没有找到有效的 src 和 tgt 字段")
        
        # 输出 jdt_map values 包含 src 的结果
        if jdt_total_count > 0:
            jdt_ratio = jdt_contains_tgt_count / jdt_total_count
            print(f"jdt_map values 包含 tgt 的比例: {jdt_ratio:.4f} ({jdt_ratio*100:.2f}%)")
            results['correct_merge_jdt_contains_tgt_ratio'] = jdt_ratio
        else:
            print("没有找到有效的 source_line 和 jdt_map 字段")
    
    return results

# 执行分析
results = analyze_ability_specific_stats(df)

# 打印汇总结果
print("\n=== 汇总结果 ===")
for key, value in results.items():
    if 'ratio' in key:
        print(f"{key}: {value:.4f} ({value*100:.2f}%)")
    else:
        print(f"{key}: {value}")

# 显示数据基本信息
print("\n=== 数据基本信息 ===")
print(df.info())
print("\n前 5 行数据:")
print(df.head())
