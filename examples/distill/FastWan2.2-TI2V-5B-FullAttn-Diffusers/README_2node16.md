# FastWan2.2 XPU 双机 16 卡测试

保留原单机脚本及 YAML 不变。新启动器通过参数覆盖为 HSDP (replicate=2, shard=8)、SP=TP=1、每卡 batch=1，全局 batch=16。
默认 5 步，禁用 W&B 和所有训练状态 checkpoint（无共享存储时不得默认启用 DCP 保存）。

## 1. 准备本地数据索引（两边均运行，CPU-only）

```bash
python examples/distill/FastWan2.2-TI2V-5B-FullAttn-Diffusers/prepare_2node_data.py \
  /home/workspace/zy_fastwan22/Wan2.2-Syn-121x704x1280_subset800 --verify-sha256
```

按 subset_manifest.json 检查文件集合、大小、行数，可选完整 SHA256；原子生成包含当前绝对路径的 file_info.pkl。
训练启用 require_existing_parquet_cache，从全局 rank0 读取并广播索引；各节点仍从自己的本地路径读 Parquet。

## 2. 启动

在两台机器的同一 Python 环境及仓库中分别启动，MASTER_ADDR/PORT 和 RUN_ID 一致。
必须由命令注入恰好八个允许使用的 CUDA_VISIBLE_DEVICES；脚本不设置或覆盖设备掩码。

```bash
# node0 / node1 分别设置 NODE_RANK=0 / 1，掩码按实际授权设置
CUDA_VISIBLE_DEVICES=<eight-permitted-device-ids> \
MASTER_ADDR=<node0-reachable-ip> NODE_RANK=<0-or-1> \
RUN_ID=wan22_dmd_2x8_smoke MAX_STEPS=5 \
bash examples/distill/FastWan2.2-TI2V-5B-FullAttn-Diffusers/distill_dmd_t2v_5B_lora_2node16.sh
```

默认 RDMA 接口为 ens21np0～ens28np0，socket/Gloo 用 ens21np0；不同环境须通过 BKCL_SOCKET_IFNAME、GLOO_SOCKET_IFNAME、BKCL_RDMA_NICS 覆盖。
其他配置仍可在命令末尾用 --dotted.key value 覆盖。50 步稳定性测试使用 MAX_STEPS=50，并使用新的 RUN_ID。
每节点日志位于 ../outputs/$RUN_ID/logs/node$NODE_RANK/。

## 3. 通信预检

preflight_2node16.py 由 torchrun --nnodes=2 --nproc_per_node=8 启动，使用相同设备掩码和 BKCL/Gloo 网络配置、独立空闲 rendezvous 端口及 DATA_DIR。
检查全局 16 rank、fp32/bf16 AllReduce、(2,8) HSDP 三步梯度解析值，以及本地数据和广播索引。
可选 EXPECTED_PCI_BUS_IDS 为允许的 PCI bus 编号十六进制逗号列表，用于断言设备范围。
所有 rank 输出 PREFLIGHT_PASS 才算通过。首次 5 步训练需覆盖 critic 和 student 更新并确认两边进程正常退出。

注意：--dry-run 仍会初始化设备、加载模型与数据，不是无副作用配置检查。

## 本地数值健康检查（可选）

设置 HEALTH_CHECK=1 会通过现有 Callback 扩展点加载本目录 dmd_health_check.DMDHealthCheck。
它检查各 rank 的有限 loss/梯度范数、teacher 冻结、critic 每步/student 每5步实际参数变化，并写本地 health/rank*.jsonl 与摘要。
会引入参数快照与额外通信开销，不适合性能基准；当前首次运行未启用该选项，不能声称已经完成数值健康检查。
