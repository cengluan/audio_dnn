
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Dataset
import os

class MyDataset(Dataset):
  def __init__(self, data_size):
    super().__init__()
    self.data = list(range(data_size))

  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, index):
    return self.data[index], index
  
def main():
  gpu_avaliable = torch.cuda.is_available()

  if gpu_avaliable:
    backend = 'nccl'
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = f'cuda: {local_rank}'
    
    torch.cuda.set_device(local_rank)
    print(f"Process {os.getpid()}: Rank {os.environ['RANK']} (Local Rank {local_rank}) using GPU {local_rank}")
  else:
    backend = 'gloo' # Gloo是cpu和跨平台兼容的后端
    local_rank = 0 # CPU模式下，local_rank通常为0,或者不严格区分
    device = 'cpu'
    print(f"Process {os.getpid()}: Rank {os.environ['RANK']} (Local Rank {os.environ['LOCAL_RANK']}) running on CPU")

  # 初始化分布式环境
  # MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE, LOCAL_RANK等环境变量由torchrun自动设置
  dist.init_process_group(backend=backend)

  # 获取全局rank和world_size
  rank = dist.get_rank()
  world_size = dist.get_world_size()

  # 3. 数据集和数据加载器
  data_size = 100
  dataset = MyDataset(data_size)

  # 4.初始化DistributedSampler
  sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)

  # 5.初始化DataLoader
  # 当使用DistributedSampler时，DataLoader的shuffle必须设置为False
  # 因为采样器已经处理了打乱逻辑
  # num_worker=0方便调试，如果数据量大且有多核cpu，可以设为大于0
  dataloader = DataLoader(dataset, batch_size=4, sampler=sampler, num_workers=0)

  # 6. 训练循环模拟
  num_epochs = 5
  for epoch in range(num_epochs):

    # 关键步骤，在每个epoch开始时调用set_epoch()
    # 这会确保DistributedSampler在该epoch使用一个新的，确定性的种子来打乱数据
    # 从而保证每个epoch的数据顺序不同但可复现
    sampler.set_epoch(epoch)

    print(f'--- Rank {rank}, Epoch {epoch} ---')

    for idx, (data, orig_idx) in enumerate(dataloader):
      # 将数据移动到相应的设备(CPU or GPU)
      data = data.to(device)

      # 只打印每一个和最后一个批次
      if idx == 0 or idx ==len(dataloader) - 1:
        print(f'Rank {rank}, Epoch {epoch}, Batch {idx}, data: {data.tolist()}, orig_idx: {orig_idx.tolist()}')

    # 确保所有进程在进入下一个Epoch前都完成了当前Epoch的数据遍历
    # 这是一个重要的同步点，防止某些进程跑得太快而其他进程还在处理上一轮数据
    dist.barrier()

  dist.destroy_process_group()
  print(f'Rank {rank} finish training and destroyed process group')

# GPU 机器
## torchrun --nproc_per_node=2 test_ddp.py

# CPU
## CUDA_VISIBLE_DEVICES="" torchrun --nproc_per_node=2 test_ddp.py
if __name__ == '__main__':
  main()