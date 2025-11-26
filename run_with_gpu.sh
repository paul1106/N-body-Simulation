#!/bin/bash

# GPU 环境设置脚本
# 用法: srun --gres=gpu:N run_with_gpu.sh ./your_program args...

# Unset 可能干扰的环境变量，让 HIP 自动检测
unset ROCR_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES

# 运行程序
exec "$@"
