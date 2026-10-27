# FastWan2.2 TI2V 5B FullAttn DMD2（XPU）

使用新版 `fastvideo.train` 入口，在单机八卡 XPU 上运行视频 DMD2 蒸馏。
student、teacher、critic 均从同一份本地权重初始化；teacher 冻结，student 和 critic 使用 LoRA r8/alpha32。

默认配置为 HSDP `(replicate=1, shard=8)`、SP=1、TP=1、每卡 batch=1、梯度累积=1，
全局 batch=8；bf16、完整梯度检查点、学习率均为 `1e-4`，运行 50 iter。
视频尺寸为 121×704×1280，DMD 时间步为 `[1000, 757, 522]`。
使用 FullAttn/FlashAttention，关闭 VSA、torch.compile、EMA 和验证视频生成。
当前框架的 LoRA 未实现 dropout，本配置不设置该参数。

## 启动

使用已安装当前仓库的可编辑 FastVideo 包及定制 PyTorch 的 uv 环境：

```bash
source /opt/venvs/python312_torch29_cuda/bin/activate
cd /home/projects/zy_fastwan22/FastVideo
bash examples/distill/FastWan2.2-TI2V-5B-FullAttn-Diffusers/distill_dmd_t2v_5B_lora.sh
```

脚本复用 `examples/train/run.sh`，显式指定八卡和 localhost rendezvous。
默认从仓库的上一级目录读取：

- 模型：`FastWan2.2-TI2V-5B-FullAttn-Diffusers/`
- 数据：`Wan2.2-Syn-121x704x1280_subset400/`，保留原 Parquet 目录层级，无需重新预处理。
- 输出：`outputs/wan22_dmd_xpu_hsdp8/`，日志位于其中的 `logs/`，W&B 默认离线。

本地 400 条样本覆盖 50 iter。`generator_update_interval=5` 表示 critic 每个 iter 更新，
student 每 5 iter 更新一次，因此 50 iter 包含 10 次 student 更新。

先运行 5 iter smoke test，覆盖 student 和 critic 更新；关闭 checkpoint 保存：

```bash
OUTPUT_DIR=/home/projects/zy_fastwan22/outputs/wan22_dmd_xpu_hsdp8_smoke \
bash examples/distill/FastWan2.2-TI2V-5B-FullAttn-Diffusers/distill_dmd_t2v_5B_lora.sh \
  --training.loop.max_train_steps 5 \
  --training.checkpoint.training_state_checkpointing_steps 0
```

可通过 `MODEL_PATH`、`DATA_DIR`、`OUTPUT_DIR`、`MASTER_PORT`、`LOG_DIR` 和 `WANDB_MODE`
覆盖启动设置；其他 YAML 参数通过脚本末尾的 `--dotted.key value` 覆盖。
`--dry-run` 仍会初始化八卡、加载模型和构建数据加载器，不只是解析配置。

八卡通信及 HSDP 小模型已验证；完整 5B DMD 训练仍需通过实际 smoke test 确认。
