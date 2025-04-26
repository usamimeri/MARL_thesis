import yaml
import numpy as np
from scipy.stats import norm
import torch
import wandb
import random
from loguru import logger
import os
from datetime import datetime


def load_config(config_path='config.yaml') -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def generate_levels(num_worker_agents) -> np.ndarray:
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
    return np.array(levels)


def distribute_evenly(total: int, max_label: int) -> np.ndarray:
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

    return np.array(result)


def inverse_weight_normalized(x: np.ndarray) -> np.ndarray:
    """
    根据输入的一维向量，计算与其值大小成反比的权重，并进行归一化处理。

    参数:
    - x: 输入的一维向量

    返回:
    - np.ndarray: 归一化后的权重向量
    """
    # 防止除以零，给定一个非常小的值
    epsilon = 1e-6
    # 计算反比权重
    weights = 1.0 / (x + epsilon)

    # 对权重进行归一化，使得权重的和为1
    normalized_weights = weights / weights.sum()

    return normalized_weights


def distribute_elements(lst, n):
    """
    将输入列表的元素尽可能均匀地分配到一个长度为 n 的numpy数组中。

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

    return np.array(result)


def gini(x: np.ndarray) -> float:
    x_sorted = np.sort(x)
    n = len(x)
    index = np.arange(1, n + 1, dtype=np.float32)
    gini = (np.sum((2 * index - n - 1) * x_sorted)) / (n * np.sum(x_sorted))
    return gini


class RunningMeanStd(object):
    """
    用于标准化输入状态，会计算每列的均值方差
    要用的时候从rms.mean和rms.var中取
    输入:batch=np.array([
    [1,2,3,4,5],
    [2,3,4,5,6],
    [3,4,5,6,7],
    [4,5,6,7,8],
    [5,6,7,8,9]
    ])
    输出：
    mean=[2.99 3.99 4.99 5.99 6.99]
    var=[2.00 2.00 2.00 2.00 2.00]

    使用：
    ```python
    rms=RunningMeanStd(shape=(5,))
    rms.update(batch)
    new_batch=(batch-rms.mean)/np.sqrt(rms.var)
    rms.update(new_batch)
    ```

    shape为状态维度
    """

    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        if isinstance(x, (float, int)):
            x = np.array([x])
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


def seed_everything(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def init_wandb():
    config = load_config()
    wandb.init(
        project="ppo-learning",
        name=f"Economic-PPO-{datetime.now().strftime('%m-%d_%H-%M')}",
        config=config,
    )


def wandb_log(name: str, pg_loss_mean, vf_loss_mean, approx_kl_mean, loss_mean):
    wandb.log({
        f"{name}/pg_loss_mean": pg_loss_mean,
        f"{name}/vf_loss_mean": vf_loss_mean,
        f"{name}/approx_kl_mean": approx_kl_mean,
        f"{name}/loss_mean": loss_mean,
    })


class Logger:
    def __init__(self, log_dir="logs", level="INFO"):
        os.makedirs(log_dir, exist_ok=True)
        self.enabled = False
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
        log_file = os.path.join(log_dir, f"{current_time}.log")
        logger.remove()
        logger.add(log_file, encoding="utf-8", level=level, colorize=True, format="{message}")
        self.logger = logger

    def info(self, message):
        if self.enabled:
            self.logger.info(message)


