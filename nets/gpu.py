
import torch

# 获取 GPU 设备
device = torch.device("cuda")

# 获取 GPU 当前内存使用情况
allocated_memory = torch.cuda.memory_allocated(device) / 1024**3  # 已分配内存，单位为 GiB
cached_memory = torch.cuda.memory_reserved(device) / 1024**3  # 保留内存，单位为 GiB
free_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3 - allocated_memory  # 总内存减去已分配内存，单位为 GiB

print(f"已分配内存: {allocated_memory:.2f} GiB")
print(f"保留内存: {cached_memory:.2f} GiB")
print(f"剩余内存: {free_memory:.2f} GiB")