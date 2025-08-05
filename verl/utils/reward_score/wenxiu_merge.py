'''
Author: yangyahe yangyahe@midu.com
Date: 2025-08-04 08:44:16
LastEditors: yangyahe yangyahe@midu.com
LastEditTime: 2025-08-05 12:24:20
FilePath: /app/yangyahe/verl/verl/utils/reward_score/wenxiu_merge.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import difflib
import random
import re

_SOLUTION_CLIP_CHARS = 512


def extract_solution(solution_str):
    # 正则表达式，要求字符串以"## 修改说明"开头，并提取"## 输出结果"之后的内容和"## 修改说明"和"## 输出结果"之间的内容
    solution_str = solution_str[:_SOLUTION_CLIP_CHARS]

    mod_marker = "## 修改说明"
    result_marker = "## 输出结果"
    pattern = rf'^{mod_marker}\s*:?\s*(.*?)\s*{result_marker}\s*:?\s*(.*)$'
    match = re.search(pattern, solution_str, re.DOTALL)  # re.DOTALL允许.匹配换行符
    if match:
        modification = match.group(1).strip()
        answer = match.group(2).strip()
        return modification, answer
    return None, None

def compute_score(solution_str, ground_truth, extra_info, format_score=0.2, score=0.9, debug=False):
    """The scoring function
    
    ​​格式分 (0.2分)​​：只要格式正确（包含"## 修改说明"和"## 输出结果"），即使答案错误
    ​​修改说明分 (0.1分)​​：修改说明长度≥10字符时获得
    ​​答案分 (0.9分)​​：答案正确时获得
    ​​最高总分 (1.0分)​​：答案正确 + 修改说明充分

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        format_score: the score for the format
        score: the score for the correct answer
    """
    
    modification, answer = extract_solution(solution_str=solution_str)
    src, tgt = extra_info["src"], ground_truth
    
    if debug or random.random() < 0.01:
        print(f"merge_score: {solution_str}, \n___src: {src}, \nanswer: {answer}, \n___tgt: {tgt}, istrue: {answer == tgt}, diff: {get_diff_details(s1=answer, s2=tgt)}")
    
    if answer is None or modification is None:
        return 0
    else:
        modi_score = 0.1 if len(modification) >= 10 else 0
        is_better = 0.1 if calculate_edit_distance_score(answer, tgt) >= calculate_edit_distance_score(src, tgt) else -0.1
        
        if answer == tgt:
            return min(1.0, score + modi_score + is_better)
        else:
            return max(0.0, format_score + modi_score + is_better)

def calculate_edit_distance_score(pred: str, target: str) -> float:
    """基于编辑距离计算相似度分数
    similarity = 2.0 * M / T
    总匹配字符数 M = len(最长公共子串) + 左侧匹配数 + 右侧匹配数
    T = len(a) + len(b)
    """
    if not pred or not target:
        return 0.0
    
    matcher = difflib.SequenceMatcher(None, pred, target)
    similarity = matcher.ratio()
    return similarity


def get_diff_details(s1, s2):
    if not s1 or not s2:
        return ["无法解析的修改说明和输出结果"]
    
    matcher = difflib.SequenceMatcher(None, s1, s2)
    result = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            result.append(f"Replace {i1}: {s1[i1:i2]} --> {s2[j1:j2]}")
        elif tag == 'delete':
            result.append(f"Delete {i1}: {s1[i1:i2]}")
        elif tag == 'insert':
            result.append(f"Insert {j1}: {s2[j1:j2]}")
        elif tag == 'equal':
            pass  # 可以忽略，也可以记录“保留了...”
    return result


if __name__ == "__main__":
    print(compute_score(solution_str="## 修改说明## 输出结果\n1", ground_truth="1\n\n2", debug=True))
    
    tests = [
        ("## 修改说明: 修复了XX问题\n## 输出结果: 成功", ("修复了XX问题", "成功")),
        ("## 修改说明 优化算法\n## 输出结果 42", ("优化算法", "42")),
        ("## 修改说明 添加新功能\n## 输出结果 已完成", ("添加新功能", "已完成")),
        ("## 修改说明: 多行内容\n第一行\n第二行\n## 输出结果: 结果文本", 
        ("多行内容\n第一行\n第二行", "结果文本")),
        ("## 修改说明内容直接写这里## 输出结果42", ("内容直接写这里", "42")),
        ("无标记文本", (None, None)),
        ("## 修改说明 只有修改说明", (None, None)),
        ("## 输出结果 只有结果", (None, None)),
        ("## 修改说明 开头有空格 \n## 输出结果 结尾有空格 ", ("开头有空格", "结尾有空格"))
    ]

    for i, (input_str, expected) in enumerate(tests):
        result = extract_solution(input_str)
        assert result == expected, f"测试{i}失败: 输入={input_str} 输出={result} 预期={expected}"
        print(f"测试{i}通过: {result}")
    
    print(calculate_edit_distance_score("你好", "好你"))
    print(calculate_edit_distance_score("你好", "你是"))
    print(calculate_edit_distance_score("自然语言", "白燃预言"))
    print(calculate_edit_distance_score("这个算法特别适合比较代码、自然语言文本等需要考虑顺序和连续性的场景，但不适合需要语义理解或长度不敏感的场合。", "这个算法特别适合比较代码、自然语言文本等需要考虑顺序和连续性的场景，但不适合需要语义理解或长度敏感的场合。"))