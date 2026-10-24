"""Quick sanity check that PyTorch sees the GPU (STUDENT_GUIDE.md Part 1.3)."""
import torch

print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA (torch)    : {torch.version.cuda}")
    print(f"cuDNN           : {torch.backends.cudnn.version()}")
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"GPU {i}           : {p.name}, {p.total_memory / 1024**3:.1f} GB")
    x = torch.randn(1024, 1024, device="cuda")
    print(f"Matmul on GPU   : OK ({(x @ x).sum().item():.2f})")
