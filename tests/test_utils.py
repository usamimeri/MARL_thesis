import pytest
from utils import distribute_evenly


def test_distribute_evenly():
    # 测试1：标签数量为 3，总数为 10
    total = 10
    max_label = 3
    result = distribute_evenly(total, max_label)

    # 每个标签至少出现一次
    for label in range(max_label):
        assert label in result, f"标签 {label} 没有出现"

    # 测试2：标签数量为 5，总数为 15
    total = 15
    max_label = 5
    result = distribute_evenly(total, max_label)

    for label in range(max_label):
        assert label in result, f"标签 {label} 没有出现"

    # 测试4：标签数量为 2，总数为 5
    total = 5
    max_label = 2
    result = distribute_evenly(total, max_label)

    # 每个标签至少出现一次
    for label in range(max_label):
        assert label in result, f"标签 {label} 没有出现"
