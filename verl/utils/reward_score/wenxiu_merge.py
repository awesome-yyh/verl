'''
Author: yangyahe yangyahe@midu.com
Date: 2025-08-04 08:44:16
LastEditors: yangyahe yangyahe@midu.com
LastEditTime: 2025-08-12 11:30:45
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
import datetime
from collections import Counter

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

def compute_score(solution_str, ground_truth, extra_info, format_score=0.1, modification_score=0.2, answer_score=0.7, debug=False):
    """The scoring function
    
    ​​格式分 (0.2分)​​：只要格式正确（包含"## 修改说明"和"## 输出结果"），即使答案错误
    ​​修改说明分 (0.1分)​​：修改说明长度≥10字符时获得
    ​​答案分 (0.7分)​​：答案正确时获得
    ​​最高总分 (1.0分)​​：答案正确 + 修改说明充分

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        format_score: the score for the format
        answer_score: the score for the correct answer
    """
    
    modification, answer = extract_solution(solution_str=solution_str)
    src, tgt = extra_info["src"], ground_truth
    
    if debug or random.random() < 0.001:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        print(f"merge_score: {solution_str}, \nsrc: {src}, \nllm: {answer}, \ntgt: {tgt}, ismodify: {answer != src}, istrue: {answer == tgt}, llm_diff: {get_diff_details(s1=src, s2=answer)}, should_diff: {get_diff_details(s1=src, s2=tgt)}, time_now: {timestamp}")
    
    if answer is None or modification is None or calculate_ngram_repetition_rate(text=modification):
        return 0.0
    
    total_score = 0.0
    
    # 1. 基础格式分
    total_score += format_score
    
    # 2. 修改说明分数
    total_score += modification_score if len(modification) >= 15 else 0
    
    # 3. 答案评分
    if answer == tgt:
        total_score += answer_score
    else:  # 答案不正确
        # 分析错误修正情况
        fixed_ratio = analyze_fixed_ratio(src, tgt, answer)
        
        # 计算相对改进度
        edit_sim = calculate_edit_distance_score(answer, tgt)
        src_sim = calculate_edit_distance_score(src, tgt)
        relative_improvement = edit_sim - src_sim
        
        # 计算答案分数（使用连续的评分）
        answer_score = answer_score * 0.5 * (  # 如果答案不正确，最多也只有0.5
            0.4 * fixed_ratio +  # 错误修正率
            0.6 * relative_improvement # 相对改进
        )
        total_score += answer_score
        
    return max(0.0, min(1.0, total_score))

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

def analyze_fixed_ratio(src: str, tgt: str, answer: str) -> float:
    """分析错误修正的比例，增加了错误类型的权重
    
    Args:
        src: 原始文本
        tgt: 目标正确文本
        answer: 模型生成的修改后文本
    
    Returns:
        float: 加权的错误修正率 (0.0-1.0)
    """
    # 错误类型权重
    ERROR_WEIGHTS = {
        'Replace': 1.0,  # 替换错误
        'Delete': 0.8,   # 删除错误
        'Insert': 0.8,   # 插入错误
    }
    
    orig_errors = get_diff_details(src, tgt)
    if not orig_errors:
        return 1.0
    
    total_weight = 0
    fixed_weight = 0
    
    for error in orig_errors:
        # 获取错误类型
        error_type = error.split()[0]
        # print(f"error: {error}, {error.split()}")
        weight = ERROR_WEIGHTS.get(error_type, 1.0)
        total_weight += weight
        
        if is_error_fixed(error, src, answer, tgt):
            fixed_weight += weight
    
    return fixed_weight / total_weight if total_weight > 0 else 0.0

def is_error_fixed(error: str, src: str, answer: str, tgt: str) -> bool:
    """检查特定错误是否被正确修复
    
    Args:
        error: 错误详情，格式如 "Replace X: old --> new" 或 "Delete X: content" 或 "Insert X: content"
        src: 原始文本
        answer: 模型生成的修改后文本
        tgt: 目标正确文本
    
    Returns:
        bool: 该错误是否被正确修复
    """
    try:
        # 解析错误类型和内容
        if error.startswith("Replace"):
            # 解析位置和替换内容
            parts = error.split(": ", 1)
            if len(parts) != 2:
                return False
                
            pos = int(parts[0].split()[1])  # 获取错误位置
            content = parts[1]
            old, new = content.split(" --> ")
            
            # 1. 检查原错误位置的内容是否被正确替换
            # 2. 检查替换后的文本是否与目标文本在该位置一致
            return (answer[pos:pos+len(new)] == new and 
                   answer[pos:pos+len(new)] == tgt[pos:pos+len(new)])
                
        elif error.startswith("Delete"):
            parts = error.split(": ", 1)
            if len(parts) != 2:
                return False
                
            pos = int(parts[0].split()[1])  # 获取删除位置
            content = parts[1]
            
            # 检查要删除的内容是否确实被删除，且周围内容正确
            context_range = 5  # 检查删除位置前后的上下文
            before_src = src[max(0, pos-context_range):pos]
            after_src = src[pos+len(content):pos+len(content)+context_range]
            
            # 在answer中找到对应上下文的位置
            try:
                context_pos = answer.index(before_src + after_src)
                return answer[context_pos:context_pos+len(before_src+after_src)] == before_src + after_src
            except ValueError:
                return False
            
        elif error.startswith("Insert"):
            parts = error.split(": ", 1)
            if len(parts) != 2:
                return False
                
            pos = int(parts[0].split()[1])  # 获取插入位置
            content = parts[1]
            
            # 检查内容是否在正确位置插入
            return (answer[pos:pos+len(content)] == content and 
                   answer[pos:pos+len(content)] == tgt[pos:pos+len(content)])
                
        return False
        
    except Exception as e:
        print(f"Error in is_error_fixed: {e}")
        return False

def calculate_ngram_repetition_rate(text: str, n: int = 4, threshold: int = 0.4) -> float:
    """计算文本的n-gram重复率"""
    tokens = list(text.strip())
    if len(tokens) < n:
        return 0.0
    
    # 生成n-gram并计数
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    counts = Counter(ngrams)
    # print(counts)
    # print((sum(counts.values()) - len(counts)) / len(ngrams))
    # 计算重复率：(总出现次数 - 独特n-gram数) / 总n-gram数
    return (sum(counts.values()) - len(counts)) / len(ngrams) > threshold
    

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

# 测试代码
def test_error_fixing():
    # 测试用例
    test_cases = [
        {
            "src": "我们一起做作业吧。",
            "tgt": "我们一起做作业吧。",
            "answer": "我们一起做作业吧。",
            "expected": True,
            "info": "原文已经正确"
        },
        {
            "src": "我们一起做作亚啊。",
            "tgt": "我们一起做作业吧。",
            "answer": "我们一起做作亚啊。",
            "expected": True,
            "info": "同一处2个错字，都没改"
        },
        {
            "src": "我们一起做作亚啊。",
            "tgt": "我们一起做作业吧。",
            "answer": "我们一起做作业啊。",
            "expected": True,
            "info": "同一处2个错字，改对1个"
        },
        {
            "src": "我们一起作作业。",
            "tgt": "我们一起做作业吧。",
            "answer": "我们一起做作业。",
            "expected": True,
            "info": "2个错字，改对1个"
        },
        {
            "src": "我们一起作作业。",
            "tgt": "我们一起做作业。",
            "answer": "我们一起做作业。",
            "expected": True,
            "info": "1个错字，改对了"
        },
        {
            "src": "我们一起作作业。",
            "tgt": "我们一起做作业。",
            "answer": "我们一起作作业。",
            "expected": True,
            "info": "1个错字，没改"
        },
        {
            "src": "这个东西地价格很贵。",
            "tgt": "这个东西的价格很贵。",
            "answer": "这个东西的价格很贵。",
            "expected": True,
            "info": "1个错字，改了"
        },
        {
            "src": "我们都在认真学习写。",
            "tgt": "我们都在认真学习。",
            "answer": "我们都在认真地学习。",
            "expected": False,
            "info": "1个错字，改了，但多了改了一处"
        },
        {
            "src": "他正在看书",
            "tgt": "他正在看书。",
            "answer": "他正在看书。",
            "expected": True,
            "info": "1个错误改了"
        }
    ]
    
    for case in test_cases:
        errors = get_diff_details(case["src"], case["tgt"])
        fixed_ratio = analyze_fixed_ratio(case["src"], case["tgt"], case["answer"])
        print(f"\nerrors: {errors}, fixed_ratio: {fixed_ratio}")
        for error in errors:
            result = is_error_fixed(error, case["src"], case["answer"], case["tgt"])
            print(f"Test case:")
            print(f"Source: {case['src']}")
            print(f"Target: {case['tgt']}")
            print(f"Answer: {case['answer']}")
            print(f"Info: {case['info']}")
            print(f"Error: {error}")
            print(f"Fixed: {result}")
        print("score:", compute_score(solution_str=f"## 修改说明 11ghjvhjjbn## 输出结果\n{case['answer']}", ground_truth=case['tgt'], extra_info={"src": case['src']}, debug=True))


if __name__ == "__main__":
    # 运行测试
    test_error_fixing()
    
    print(compute_score(solution_str="## 修改说明## 输出结果\n1", ground_truth="2", extra_info={"src": "1"}, debug=True))
    
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