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

  
def apply_edits(original: str, prediction: str) -> str | None:
    """根据编辑操作生成修正后的句子
    
    参数:
        original: 原始文本
        prediction: 模型预测的编辑指令
        
    返回:
        str: 修正后的文本
        None: 无效的编辑操作
    """
    NOOP_RE = re.compile(r'^\s*<noop\s*/>\s*$', flags=re.IGNORECASE)
    EDIT_PAIR_RE = re.compile(
        r'^\s*(?:<error>(.*?)</error>\s*<correct>(.*?)</correct>\s*)+$', 
        re.DOTALL | re.IGNORECASE
    )
    
    stripped_pred = prediction.strip()
    # 处理无操作指令
    if NOOP_RE.fullmatch(stripped_pred):
        return original
    
    if not EDIT_PAIR_RE.fullmatch(stripped_pred):
        return None
    
    # 解析编辑对并过滤空错误
    pairs = []
    for match in re.finditer(r'<error>(.*?)</error>\s*<correct>(.*?)</correct>', stripped_pred, re.DOTALL):
        err, cor = match.groups()
        if not err:  # 空错误无效
            return None
        pairs.append((err, cor))
    
    if not pairs:
        return None
    
    # 一次性收集所有错误位置
    spans = []
    for err, _ in pairs:
        start = original.find(err)
        if start == -1 or original.find(err, start + 1) != -1:
            return None  # 未找到或非唯一
        spans.append((start, start + len(err)))
    
    # 检查重叠
    spans.sort()
    if any(end > next_start for (_, end), (next_start, _) in zip(spans, spans[1:])):
        return None
    
    # 构建修正映射（允许相同错误不同修正）
    corrections = {e: c for e, c in pairs}
    
    # 应用替换
    parts, last_end = [], 0
    for (start, end), (err, _) in zip(spans, pairs):
        parts += [original[last_end:start], corrections.get(err, err)]
        last_end = end
    parts.append(original[last_end:])
    
    return ''.join(parts)

def compute_score(solution_str, ground_truth, extra_info, format_score=0.1, answer_score=0.9, debug=False):
    """The scoring function
    
    ​​格式分 (0.1分)​​：只要格式正确（包含"## 修改说明"和"## 输出结果"），即使答案错误
    ​​修改说明分 (0.2分)​​：修改说明长度≥15字符时获得基础分，包含原词和建议词时获得更高分
    ​​答案分 (0.7分)​​：答案正确时获得
    ​​最高总分 (1.0分)​​：答案正确 + 修改说明充分

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        format_score: the score for the format
        answer_score: the score for the correct answer
    """
    solution_str = solution_str[:_SOLUTION_CLIP_CHARS].rpartition("</think>")[2].strip()
    src, tgt = extra_info["src"], ground_truth
    answer = apply_edits(original=src, prediction=solution_str)
    
    if debug or random.random() < 0.001:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        print(f"solution_str: {solution_str}, \nsrc: {src}, \nllm: {answer}, \ntgt: {tgt}, ismodify: {answer != src}, istrue: {answer == tgt}, llm_diff: {get_diff_details(s1=src, s2=answer)}, should_diff: {get_diff_details(s1=src, s2=tgt)}, time_now: {timestamp}")
    
    if answer is None:
        return 0.0
    
    total_score = 0.0
    
    # 1. 基础格式分
    total_score += format_score
    
    # 3. 答案评分
    if answer == tgt:
        total_score += answer_score
    else:  # 答案不正确
        # 分析错误修正情况
        fixed_ratio, detected_ratio = analyze_fixed_ratio(src, tgt, answer)
        
        # 计算相对改进度
        edit_sim = calculate_edit_distance_score(answer, tgt)
        src_sim = calculate_edit_distance_score(src, tgt)
        relative_improvement = max(0, edit_sim - src_sim)  # 确保改进值非负
        
        # 计算答案分数（使用连续的评分）
        answer_score = answer_score * 0.8 * (  # 如果答案不正确，最多也只有0.8
            0.3 * detected_ratio + # 错误检测率
            0.4 * fixed_ratio +  # 错误修正率
            0.3 * relative_improvement # 相对改进
        )
        
        total_score += answer_score
    
    if debug or random.random() < 0.01:
        print(f"== total_score: {total_score}, solution_str: {solution_str}, \nsrc: {src}, \nllm: {answer}, \ntgt: {tgt}, ismodify: {answer != src}, istrue: {answer == tgt}, llm_diff: {get_diff_details(s1=src, s2=answer)}, should_diff: {get_diff_details(s1=src, s2=tgt)}")
    
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
        return (1.0, 1.0) if answer == tgt else (0.0, 0.0)
    
    total_weight = 0
    fixed_weight = 0
    detected_weight = 0
    
    for error in orig_errors:
        # 获取错误类型
        error_type = error.split()[0]
        # print(f"error: {error}, {error.split()}")
        weight = ERROR_WEIGHTS.get(error_type, 1.0)
        total_weight += weight
        
        if is_error_fixed(error, src, answer, tgt):
            fixed_weight += weight
        if is_error_detected(error, src, answer, tgt):
            detected_weight += weight
    
    return (fixed_weight / total_weight, detected_weight / total_weight) if total_weight > 0 else (0.0, 0.0)

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
            # 检查目标错误是否被正确修复
            # 使用更可靠的方法：对比前后内容
            context_len = min(5, pos)  # 使用上下文来确保位置匹配
            before_ctx = src[max(0, pos-context_len):pos]
            
            # 在answer中找到对应上下文位置
            try:
                ctx_pos = answer.find(before_ctx)
                if ctx_pos == -1:
                    return False
                    
                fix_pos = ctx_pos + len(before_ctx)
                # 检查替换后的内容是否正确
                return answer[fix_pos:fix_pos+len(new)] == new
            except Exception:
                return False
                
                
        elif error.startswith("Delete"):
            parts = error.split(": ", 1)
            if len(parts) != 2:
                return False
                
            pos = int(parts[0].split()[1])  # 获取删除位置
            content = parts[1]
            
            # 检查要删除的内容是否确实被删除，且周围内容正确
            context_len = min(5, pos)  # 使用更少的上下文以增加匹配概率
            before_src = src[max(0, pos-context_len):pos]
            after_src = src[pos+len(content):pos+len(content)+context_len]
            
            # 在answer中找到对应上下文的位置
            try:
                # 查找前后上下文的连接位置
                connected = before_src + after_src
                ctx_pos = answer.find(connected)
                
                # 如果找到了连接的上下文，说明内容被删除了
                return ctx_pos != -1
            except Exception:
                return False
            
        elif error.startswith("Insert"):
            parts = error.split(": ", 1)
            if len(parts) != 2:
                return False
                
            pos = int(parts[0].split()[1])  # 获取插入位置
            content = parts[1]
            
            # 检查内容是否在正确位置插入
            # return (answer[pos:pos+len(content)] == content == tgt[pos:pos+len(content)])
            # 为了更准确地定位插入位置，使用上下文
            context_len = min(5, pos)
            before_ctx = src[max(0, pos-context_len):pos]
            after_ctx = src[pos:pos+context_len]
            
            # 在answer中找对应位置
            try:
                before_pos = answer.find(before_ctx)
                if before_pos == -1:
                    return False
                    
                insert_pos = before_pos + len(before_ctx)
                # 检查插入的内容是否正确
                return answer[insert_pos:insert_pos+len(content)] == content
            except Exception:
                return False
                
        return False
        
    except Exception as e:
        print(f"Error in is_error_fixed: {e}")
        return False

def is_error_detected(error: str, src: str, answer: str, tgt: str) -> bool:
    """检查错误位置是否被识别和修改（不关心修改结果是否正确）
    
    Args:
        error: 错误详情，格式如 "Replace X: old --> new" 或 "Delete X: content" 或 "Insert X: content"
        src: 原始文本
        answer: 模型生成的修改后文本
        tgt: 目标正确文本（不再使用）
    
    Returns:
        bool: 错误位置是否被修改（即错误被发现）
    """
    try:
        # 改进检测逻辑，使用上下文确定位置
        if error.startswith("Replace"):
            parts = error.split(": ", 1)
            if len(parts) != 2:
                return False
                
            pos = int(parts[0].split()[1])
            content = parts[1]
            old, _ = content.split(" --> ")
            
            # 使用上下文确定位置
            context_len = min(5, pos)
            before_ctx = src[max(0, pos-context_len):pos]
            
            try:
                ctx_pos = answer.find(before_ctx)
                if ctx_pos == -1:
                    return True  # 上下文也变了，说明有修改
                    
                check_pos = ctx_pos + len(before_ctx)
                # 只要这个位置的内容发生了变化，就认为错误被检测到
                return check_pos >= len(answer) or answer[check_pos:check_pos+len(old)] != old
            except Exception:
                return False
                
        elif error.startswith("Delete"):
            parts = error.split(": ", 1)
            if len(parts) != 2:
                return False
                
            pos = int(parts[0].split()[1])
            content = parts[1]
            
            # 使用上下文
            context_len = min(5, pos)
            before_ctx = src[max(0, pos-context_len):pos]
            after_ctx = src[pos+len(content):pos+len(content)+context_len]
            
            # 检查原内容是否被删除
            try:
                # 如果能找到前后上下文的连接，说明内容被删除了
                connected = before_ctx + after_ctx
                return answer.find(connected) != -1
            except Exception:
                return False
            
        elif error.startswith("Insert"):
            parts = error.split(": ", 1)
            if len(parts) != 2:
                return False
                
            pos = int(parts[0].split()[1])
            
            # 使用上下文
            context_len = min(5, pos)
            before_ctx = src[max(0, pos-context_len):pos]
            after_ctx = src[pos:pos+context_len]
            
            # 检查是否在此位置有插入
            try:
                before_pos = answer.find(before_ctx)
                if before_pos == -1:
                    return True  # 上下文也变了，说明有修改
                    
                after_pos = answer.find(after_ctx, before_pos + len(before_ctx))
                # 如果前后上下文距离变长，说明有内容被插入
                return after_pos > before_pos + len(before_ctx)
            except Exception:
                return False
                
        return False
        
    except Exception as e:
        print(f"Error in is_error_detected: {e}")
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
        },
        {
            "src": "他不是正在看书",
            "tgt": "他正在看书。",
            "answer": "他正在看树。",
            "expected": True,
            "info": "1个错误改了"
        }
    ]
    
    for case in test_cases:
        errors = get_diff_details(case["src"], case["tgt"])
        fixed_ratio, detected_ratio = analyze_fixed_ratio(case["src"], case["tgt"], case["answer"])
        print(f"\nerrors: {errors}, fixed_ratio: {fixed_ratio}, detected_ratio: {detected_ratio}")
        for error in errors:
            print(f"Test case:")
            print(f"Source: {case['src']}")
            print(f"Target: {case['tgt']}")
            print(f"Answer: {case['answer']}")
            print(f"Info: {case['info']}")
            print(f"Error: {error}")
            result = is_error_fixed(error, case["src"], case["answer"], case["tgt"])
            print(f"=== Fixed: {result}")
            result = is_error_detected(error, case["src"], case["answer"], case["tgt"])
            print(f"=== Fixed: {result}")
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