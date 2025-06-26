import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

def setup(rank, world_size):
    # 初始化分布式环境
    dist.init_process_group(
        backend='nccl',  # GPU 推荐用 nccl
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank % torch.cuda.device_count())

def cleanup():
    dist.destroy_process_group()

def demo_basic(rank, world_size):
    setup(rank, world_size)
    # 构造简单模型和数据
    model = torch.nn.Linear(10, 1).cuda()
    ddp_model = DDP(model, device_ids=[rank % torch.cuda.device_count()])
    dataset = TensorDataset(torch.randn(100, 10), torch.randn(100, 1))
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=8, sampler=sampler)
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()
    for epoch in range(5):
        sampler.set_epoch(epoch)
        for x, y in dataloader:
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            output = ddp_model(x)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
        if rank == 0:
            print(f"Epoch {epoch} finished.")
    cleanup()

def main():
    world_size = int(os.environ['WORLD_SIZE'])  # 总进程数 = 节点数 * 每节点GPU数
    rank = int(os.environ['RANK'])              # 当前进程的全局rank
    demo_basic(rank, world_size)

if __name__ == "__main__":
    main() 