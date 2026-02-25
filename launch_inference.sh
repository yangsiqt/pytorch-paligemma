#!/bin/bash
# launch_inference.sh - PaliGemma 推理启动脚本
# 配置模型路径、prompt、图像、生成参数等，调用 inference.py

# 模型权重目录（需包含 config.json 和 *.safetensors）
MODEL_PATH="$HOME/projects/paligemma-weights/paligemma-3b-pt-224"
# 图文条件生成时的文本 prompt（模型将根据图像续写）
PROMPT="this building is "
# 输入图像路径
IMAGE_FILE_PATH="test_images/pic1.jpeg"
# 最大生成 token 数
MAX_TOKENS_TO_GENERATE=100
# 采样温度（越高越随机，越低越确定）
TEMPERATURE=0.8
# Top-p (nucleus) 采样参数，累积概率超过 p 的 token 之外的将被剔除
TOP_P=0.9
# 是否使用随机采样（False 则为 greedy 取最大概率）
DO_SAMPLE="False"
# 是否强制只用 CPU（True 时忽略 CUDA/MPS）
ONLY_CPU="False"

# 调用推理脚本，将上述变量作为命令行参数传入
python inference.py \
    --model_path "$MODEL_PATH" \
    --prompt "$PROMPT" \
    --image_file_path "$IMAGE_FILE_PATH" \
    --max_tokens_to_generate $MAX_TOKENS_TO_GENERATE \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --do_sample $DO_SAMPLE \
    --only_cpu $ONLY_CPU \

