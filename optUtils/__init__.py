#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/6/3 16:25
# @Author : LYX-夜光

import codecs
import json
import os
import random
import yaml
import numpy as np

import warnings
warnings.filterwarnings("ignore")


# 创建文件夹
def make_dirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)

# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

# 写入json文件
def write_json(jsonPath, value):
    with open(jsonPath, "w", encoding="utf-8") as f:
        f.write(json.dumps(value))

# 获取json文件
def read_json(jsonPath):
    with open(jsonPath, "r", encoding="utf-8") as f:
        value = json.loads(f.read())
    return value

# 读取yaml配置文件
def read_yaml(yamlFile):
    with codecs.open(yamlFile, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

yaml_config = read_yaml(os.path.join(os.path.dirname(__file__), '../param.yaml'))
