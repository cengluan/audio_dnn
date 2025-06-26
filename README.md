# PyTorch 多机多卡分布式训练示例

本项目提供了一个基于 PyTorch 的多机多卡分布式训练（DistributedDataParallel, DDP）模板，适用于多台机器、每台多张 GPU 的场景。

## 1. 代码说明

- `train_ddp.py`：分布式训练主脚本，包含模型、数据、分布式初始化等基本流程。
- 可根据实际需求替换模型和数据部分。

## 2. 启动方式

假设有 2 台机器，每台 4 张 GPU：
- master 节点 IP: `192.168.1.1`
- node2 节点 IP: `192.168.1.2`

### 使用 torchrun（推荐 PyTorch 1.9+）

**master 节点：**
```bash
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=192.168.1.1 --master_port=12345 train_ddp.py
```

**node2 节点：**
```bash
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr=192.168.1.1 --master_port=12345 train_ddp.py
```

### 使用 torch.distributed.launch

**master 节点：**
```bash
MASTER_ADDR=192.168.1.1 MASTER_PORT=12345 WORLD_SIZE=8 RANK=0 \
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=192.168.1.1 --master_port=12345 train_ddp.py
```

**node2 节点：**
```bash
MASTER_ADDR=192.168.1.1 MASTER_PORT=12345 WORLD_SIZE=8 RANK=4 \
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr=192.168.1.1 --master_port=12345 train_ddp.py
```

## 3. 参数说明
- `--nproc_per_node`：每台机器的 GPU 数量。
- `--nnodes`：总机器数。
- `--node_rank`：当前机器编号（从 0 开始）。
- `--master_addr`：主节点 IP。
- `--master_port`：主节点端口。

## 4. SSH 免密登录配置（多机分布式必需）

### 为什么需要 SSH 免密？
- 分布式训练时，主节点需要通过 SSH 自动启动其他节点的训练进程。
- 如果没有配置免密 SSH，主节点在启动其他节点进程时会卡在输入密码，导致训练无法自动化进行。
- 配置 SSH 免密后，训练脚本可以一键启动所有节点，无需人工干预。

### 配置步骤
1. **在主节点生成 SSH 密钥对**（如果还没有）：
   ```bash
   ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
   ```
   一路回车即可。

2. **将公钥分发到所有工作节点**（假设用户名为 user，节点 IP 为 192.168.1.2）：
   ```bash
   ssh-copy-id user@192.168.1.2
   ```
   按提示输入一次密码即可。

3. **测试免密登录**：
   ```bash
   ssh user@192.168.1.2
   ```
   如果能直接登录且无需输入密码，则配置成功。

4. **对所有节点都执行上述操作**，保证主节点能免密 SSH 到所有工作节点。

> 只需主节点能免密登录到所有工作节点即可，不要求节点之间互相免密。

## 5. 其他说明
- 推荐使用 NCCL 后端（已在代码中设置）。
- 代码中的模型、数据集和训练循环可根据实际需求替换。
- 启动命令中的参数需根据实际集群环境调整。

如需集成自定义模型或数据，或有其他分布式训练相关问题，欢迎交流！ 