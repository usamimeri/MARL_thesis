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
    max_value = 2.0           # 技能禀赋最大值
    min_percentile = norm.cdf(min_value)
    # 构造均匀的分位点
    max_percentile = norm.cdf(max_value)
    percentiles = np.linspace(min_percentile, max_percentile, num_worker_agents)
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


def inverse_weight_normalized(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    根据输入的一维张量，计算与其值大小成反比的权重，并进行归一化处理。

    参数:
    - input_tensor (torch.Tensor): 输入的一维张量

    返回:
    - torch.Tensor: 归一化后的权重向量
    """
    # 防止除以零，给定一个非常小的值
    epsilon = 1e-6
    # 计算反比权重
    weights = 1.0 / (input_tensor + epsilon)

    # 对权重进行归一化，使得权重的和为1
    normalized_weights = weights / weights.sum()

    return normalized_weights


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


def gini(x: torch.Tensor) -> float:
    x_sorted, _ = torch.sort(x)
    n = len(x)
    index = torch.arange(1, n + 1, dtype=torch.float32, device=x.device)
    gini = (torch.sum((2 * index - n - 1) * x_sorted)) / (n * torch.sum(x_sorted))
    return gini


class RunningMeanStd(torch.nn.Module):
    def __init__(self, shape: tuple):
        super(RunningMeanStd, self).__init__()
        self.mean = torch.zeros(shape, dtype=torch.float64)  # 每列的均值
        self.var = torch.ones(shape, dtype=torch.float64)    # 每列的方差，初始化为1
        self.count = 1e-4                                 # 初始化计数，防止第一次除零

    def update(self, x):
        batch_mean = x.mean(dim=0)  # 计算输入x在第一个维度（batch维度）上的均值
        batch_var = x.var(dim=0, unbiased=False)  # 计算输入x在第一个维度上的方差，unbiased=False时为样本方差
        batch_count = x.size(0)  # 获取x中的数据点数量（即batch大小）
        self.update_from_moments(batch_mean, batch_var, batch_count)  # 更新均值和方差

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        # 使用增量更新公式更新mean, var, count
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean  # 新旧均值的差
    tot_count = count + batch_count  # 总数据点数

    # 更新均值
    new_mean = mean + delta * batch_count / tot_count

    # 更新方差
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + (delta ** 2) * count * batch_count / tot_count  # 计算M2
    new_var = M2 / tot_count  # 归一化方差
    new_count = tot_count  # 更新总数据点数

    return new_mean, new_var, new_count
