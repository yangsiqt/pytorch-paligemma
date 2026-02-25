"""
inference.py - PaliGemma 推理脚本

本文件实现图文条件生成的推理流程：
- move_inputs_to_device: 将输入张量移到指定设备
- get_model_inputs: 加载图像、构造 prompt，得到模型输入
- test_inference: 自回归生成循环（含 KV Cache、Top-p 采样等）
- _sample_top_p: Nucleus (Top-p) 采样
- main: 命令行入口，加载模型并运行推理
"""

from PIL import Image
import torch
import fire

from processing_paligemma import PaliGemmaProcessor
from modeling_gemma import KVCache, PaliGemmaForConditionalGeneration
from utils import load_hf_model


def move_inputs_to_device(model_inputs: dict, device: str):
    """将模型输入字典中的张量全部移动到指定设备（CPU/CUDA/MPS）"""
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}  # 遍历并移至 device
    return model_inputs


def get_model_inputs(
    processor: PaliGemmaProcessor, prompt: str, image_file_path: str, device: str
):
    """根据 prompt 和图像路径构造模型输入（input_ids、attention_mask、pixel_values）并移至设备"""
    image = Image.open(image_file_path)  # 打开图像文件
    images = [image]
    prompts = [prompt]
    model_inputs = processor(text=prompts, images=images)  # 预处理：tokenize + 图像处理
    model_inputs = move_inputs_to_device(model_inputs, device)
    return model_inputs


def test_inference(
    model: PaliGemmaForConditionalGeneration,
    processor: PaliGemmaProcessor,
    device: str,
    prompt: str,
    image_file_path: str,
    max_tokens_to_generate: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
):
    """执行自回归生成：循环调用模型，每次生成一个 token，直到遇到 EOS 或达到最大长度"""
    model_inputs = get_model_inputs(processor, prompt, image_file_path, device)
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    pixel_values = model_inputs["pixel_values"]

    kv_cache = KVCache()  # 初始化 KV 缓存，用于加速自回归生成

    # Generate tokens until you see the stop token
    # 持续生成直到遇到结束 token
    stop_token = processor.tokenizer.eos_token_id
    generated_tokens = []

    for _ in range(max_tokens_to_generate):
        # Get the model outputs
        # TODO: remove the labels
        # 获取模型输出
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )
        kv_cache = outputs["kv_cache"]  # 更新缓存供下一轮使用
        next_token_logits = outputs["logits"][:, -1, :]  # 取最后一个位置的 logits
        # Sample the next token
        # 根据 do_sample 选择 greedy 或 采样
        if do_sample:
            # Apply temperature
            # 温度缩放后 softmax，temperature 越高分布越平坦
            next_token_logits = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token = _sample_top_p(next_token_logits, top_p)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # greedy 取概率最大
        assert next_token.size() == (1, 1)
        next_token = next_token.squeeze(0)  # Remove batch dimension 去掉 batch 维
        generated_tokens.append(next_token)
        # Stop if the stop token has been generated
        # 若生成结束 token 则停止
        if next_token.item() == stop_token:
            break
        # Append the next token to the input
        # 将新 token 拼接到 input，作为下一轮输入
        input_ids = next_token.unsqueeze(-1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), device=input_ids.device)], dim=-1
        )

    generated_tokens = torch.cat(generated_tokens, dim=-1)  # 沿序列维拼接
    # Decode the generated tokens
    # 将 token id 解码为字符串
    decoded = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    print(prompt + decoded)


def _sample_top_p(probs: torch.Tensor, p: float):
    """Nucleus (Top-p) 采样：按概率从累积概率不超过 p 的 token 集合中采样"""
    # (B, vocab_size)
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)  # 按概率降序排列
    # (B, vocab_size)
    probs_sum = torch.cumsum(probs_sort, dim=-1)  # 累积和
    # (B, vocab_size)
    # (Substracting "probs_sort" shifts the cumulative sum by 1 position to the right before masking)
    # 减去 probs_sort 使累积和在掩码前右移一位
    mask = probs_sum - probs_sort > p
    # Zero out all the probabilities of tokens that are not selected by the Top P
    # 将不在 top-p 内的 token 概率置零
    probs_sort[mask] = 0.0
    # Redistribute the probabilities so that they sum up to 1.
    # 重新归一化使概率和为 1
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    # Sample a token (its index) from the top p distribution
    # 从 top-p 分布中采样
    next_token = torch.multinomial(probs_sort, num_samples=1)
    # Get the token position in the vocabulary corresponding to the sampled index
    # 用 probs_idx 映射回原始词表中的 token id
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def main(
    model_path: str = None,
    prompt: str = None,
    image_file_path: str = None,
    max_tokens_to_generate: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.9,
    do_sample: bool = False,
    only_cpu: bool = False,
):
    """命令行入口：解析参数、加载模型、运行推理。使用 fire.Fire 自动解析命令行。"""
    device = "cpu"

    if not only_cpu:
        if torch.cuda.is_available():
            device = "cuda"   # 优先 GPU
        elif torch.backends.mps.is_available():
            device = "mps"    # Apple Silicon 用 MPS

    print("Device in use: ", device)

    print(f"Loading model")
    model, tokenizer = load_hf_model(model_path, device)
    model = model.to(device).eval()  # 推理模式

    num_image_tokens = model.config.vision_config.num_image_tokens  # 图像 token 数量
    image_size = model.config.vision_config.image_size  # 输入图像尺寸
    processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)

    print("Running inference")
    with torch.no_grad():  # 推理时不计算梯度
        test_inference(
            model,
            processor,
            device,
            prompt,
            image_file_path,
            max_tokens_to_generate,
            temperature,
            top_p,
            do_sample,
        )


if __name__ == "__main__":
    fire.Fire(main)
