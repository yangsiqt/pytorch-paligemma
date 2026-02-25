"""
utils.py - 模型加载工具

从本地目录加载 PaliGemma 模型：读取 config.json、safetensors 权重，
构建 PaliGemmaForConditionalGeneration 并加载参数，同时加载 tokenizer。
"""

from modeling_gemma import PaliGemmaForConditionalGeneration, PaliGemmaConfig
from transformers import AutoTokenizer
import json
import glob
from safetensors import safe_open
from typing import Tuple
import os


def load_hf_model(model_path: str, device: str) -> Tuple[PaliGemmaForConditionalGeneration, AutoTokenizer]:
    """从 model_path 加载 tokenizer 与 PaliGemma 模型，返回 (model, tokenizer)"""
    # Load the tokenizer
    # 加载 tokenizer，右填充与自回归生成匹配
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    assert tokenizer.padding_side == "right"

    # Find all the *.safetensors files
    # 查找所有 safetensors 权重文件
    safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))

    # ... and load them one by one in the tensors dictionary
    # 逐个加载到 tensors 字典
    tensors = {}
    for safetensors_file in safetensors_files:
        with safe_open(safetensors_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    # Load the model's config
    # 加载 config.json
    with open(os.path.join(model_path, "config.json"), "r") as f:
        model_config_file = json.load(f)
        config = PaliGemmaConfig(**model_config_file)

    # Create the model using the configuration
    # 按配置创建模型
    model = PaliGemmaForConditionalGeneration(config).to(device)

    # Load the state dict of the model
    # 加载权重；strict=False 以兼容多余/缺失的 key
    model.load_state_dict(tensors, strict=False)

    # Tie weights
    # 绑定 embed 与 lm_head 权重
    model.tie_weights()

    return (model, tokenizer)