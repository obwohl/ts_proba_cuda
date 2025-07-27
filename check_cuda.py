import torch
print(f"PyTorch version: {torch.__version__}")
if not torch.cuda.is_available():
    print("🚨 FEHLER: CUDA ist immer noch nicht verfügbar!")
else:
    print("✅ CUDA ist verfügbar!")
    print(f"Installierte CUDA-Version von PyTorch: {torch.version.cuda}")
    print(f"GPU-Modell: {torch.cuda.get_device_name(0)}")
    major, minor = torch.cuda.get_device_capability(0)
    print(f"Compute Capability: sm_{major}{minor}")
