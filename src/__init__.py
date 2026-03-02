"""
大模型学习指南 - 源代码模块

本目录包含可复用的代码模块，供Notebook中使用。
"""

__version__ = "0.1.0"
__author__ = "LLM Learning Guide Team"

# 导入常用模块
from . import models
from . import utils
from . import data
from . import inference

__all__ = [
    'models',
    'utils',
    'data',
    'inference',
]
