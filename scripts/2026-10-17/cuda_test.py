import time, torch
print("torch", torch.__version__, "cuda rt", torch.version.cuda, torch.cuda.get_device_name(0))
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize()
t = time.time()
for _ in range(50):
    y = x @ x
torch.cuda.synchronize()
print(f"matmul x50: {time.time()-t:.3f}s")
a = torch.ones(1000, dtype=torch.bool, device="cuda")
t = time.time()
for i in range(2000):
    a = a & torch.ones((), dtype=torch.bool, device="cuda")
torch.cuda.synchronize()
print(f"2000 small kernels: {time.time()-t:.3f}s")
