import argparse
import os
from tqdm import tqdm
import ast
import json
import random
import pandas as pd
from datasets import Dataset
from typing import List, Dict, Tuple, Optional


def parse_line(line: str) -> Tuple[str, str]:
    """
    解析单行文本，支持多种格式：
    1. JSON格式：{"src": "...", "tgt": "..."}
    2. 分隔符格式：问题\t答案
    """
    line = line.strip()
    if not line:
        return None
    
    try:
        sents_map = ast.literal_eval(line)

        src = sents_map.pop("sentence")
        tgt = sents_map.pop("corr_sentence")
        preds_map = sents_map
        
        return src, tgt, preds_map
    except Exception as e:
        print(f"解析错误: {line}")
        return None
    

def txt_to_parquet(
    input_path: str,
    output_path: str,
    data_source: str = "custom",
    ability: str = "general",
    instruction: str = "Let's think step by step and output the final tgt after \"####\".",
    split: str = "train"
):
    """
    将TXT文件转换为Parquet格式
    
    参数:
        input_path: 输入TXT文件路径
        output_path: 输出Parquet文件路径
        data_source: 数据来源标识
        ability: 数据类型（math, commonsense, code等）
        instruction: 系统提示词
        split: 数据集划分（train/validation/test）
    """
    # 读取并解析文件
    instruction = """你是校对助手，针对提供的原句和多个不同系统给出的参考句子，参考结果不一定正确，需要你综合判断。
原句无错就不要改动，改动尽量接近原句，让我们一步一步思考，最后输出最为合理的结果。例如：
## 原句
自从楼下开了瑞幸，我喝咖啡是极其妨碍的了。

## 参考修改句
参考修改1：自从楼下开了瑞幸，我喝咖啡是极其频繁的了。
参考修改2：自从楼下开了瑞幸，我喝咖啡是极其方便的了。

## 修改说明
原句中“妨碍”使用不当，“妨碍”指干扰、阻碍，使事情不能顺利进行，而此处想表达的应该是喝咖啡变得很便利，所以可以改为“方便”或“频繁”等词。

## 输出结果
自从楼下开了瑞幸，我喝咖啡是极其方便的了。
"""
    cor_in_preds, count_t2t, count_all = 0, 0, 0
    samples = []
    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in tqdm(enumerate(f), total=3211817):
            try:
                if split == "test" and random.random() > 1/787*3:
                    continue
                
                parsed = parse_line(line)
                if parsed:
                    src, tgt, references = parsed
                    if tgt in references.values():
                        if random.random() > 0.14:
                            continue
                        cor_in_preds += 1
                    
                    if src == tgt:
                        count_t2t += 1                    
                    
                    count_all += 1

                    user_input = f"## 原句\n{src}"
                    
                    if references:
                        references = [references[k] for k in sorted(references.keys())]
                        ref_text = "\n".join(
                            [f"参考修改{idx}：{c}" for idx, c in enumerate(dict.fromkeys(references), 1)]  
                        )
                        user_input += f"\n\n## 参考修改句\n{ref_text}"
                    else:
                        user_input += f"\n\n## 参考修改句\n无参考修改句"
                    
                    samples.append({
                        "data_source": data_source,
                        "prompt": [
                            {"role": "system", "content": instruction},
                            {"role": "user", "content": f"{user_input}"}
                        ],
                        "ability": ability,
                        "reward_model": {"style": "rule", "ground_truth": f"{src}\n\n{tgt}"},
                        "extra_info": {
                            "split": split,
                            "index": i,
                            "tgt": tgt,
                            "src": src,
                            "source_line": line.strip()
                        }
                    })
            except Exception as e:
                print(f"跳过第 {i+1} 行 (错误: {str(e)}): {line.strip()}")
    
    if not samples:
        raise ValueError("未解析出任何有效数据")
    
    print(f"成功解析 {len(samples)} 条样本")
    
    # 转换为Parquet
    df = pd.DataFrame(samples)
    dataset = Dataset.from_pandas(df)
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存为Parquet
    dataset.to_parquet(output_path)
    print(f"已保存到: {output_path}")
    print(f"包含正确句的比例: {cor_in_preds} / {count_all} = {cor_in_preds/count_all:.2f}")
    print(f"正确到正确的比例: {count_t2t} / {count_all} = {count_t2t/count_all:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将TXT文件转换为强化学习训练用的Parquet格式")
    parser.add_argument("--input", required=True, help="输入TXT文件路径")
    parser.add_argument("--output", required=True, help="输出Parquet文件路径")
    parser.add_argument("--data_source", required=True, help="数据来源标识")
    parser.add_argument("--ability", default="correct_merge", help="数据类型 (math/code/commonsense等)")
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"], help="数据集划分")
    
    args = parser.parse_args()
    
    txt_to_parquet(
        input_path=args.input,
        output_path=args.output,
        data_source=args.data_source,
        ability=args.ability,
        split=args.split
    )