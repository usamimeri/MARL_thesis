from typing import List
import yaml
import numpy as np
from scipy.stats import norm
import torch


def load_config(config_path='config.yaml') -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def generate_levels(num_worker_agents) -> list:
    """利用标准正态分布初始化技能禀赋"""
    min_value = 0.5           # 技能禀赋最小值
    num_worker_agents           # 采样点数量
    min_percentile = norm.cdf(min_value)
    # 构造均匀的分位点
    percentiles = np.linspace(min_percentile, 0.9999, num_worker_agents)
    levels = norm.ppf(percentiles)
    # 随机打乱
    np.random.shuffle(levels)
    return levels


def distribute_evenly(total: int, max_label: int) -> List[int]:
    """
    将 total 个元素均匀分配到标签 0,1,...,max_label-1 共 max_label 个桶中。
    每个标签至少出现一次，且各标签出现次数尽可能均匀。
    用于初始化每个工人的企业，输出为企业对应的索引
    """
    # 标签总数
    num_buckets = max_label

    # 基础分配：每个桶至少 base_cnt 个
    base_cnt, remainder = divmod(total, num_buckets)
    # remainder 个桶多取 1 个，保证总和为 total
    counts = [base_cnt + (1 if i < remainder else 0) for i in range(num_buckets)]

    result = []
    for label, cnt in enumerate(counts):
        result.extend([label] * cnt)

    np.random.shuffle(result)

    return result


def distribute_elements(lst, n):
    """
    将输入列表的元素尽可能均匀地分配到一个长度为 n 的列表中。

    参数:
        lst (list): 输入的列表，其中包含需要分配的元素。
        n (int): 目标列表的长度。

    返回:
        list: 长度为 n 的列表，元素尽可能均匀地分配。

    例子:
        输入:
            lst = [1, 10, 20]
            n = 10
        输出:
            [1, 10, 20, 1, 10, 20, 1, 10, 20, 1]
    """

    # 计算总元素个数
    total_elements = len(lst)

    # 计算每个元素应出现的最小次数
    base_count = n // total_elements

    # 计算剩余的元素数
    remainder = n % total_elements

    # 初始化结果列表
    result = []

    # 为每个元素添加 base_count 次
    for i in range(total_elements):
        result.extend([lst[i]] * base_count)

    # 为剩余的元素添加 1 次
    for i in range(remainder):
        result.append(lst[i])

    np.random.shuffle(result)

    return result



