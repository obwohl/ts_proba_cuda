
import torch

if torch.cuda.is_available():
    print("CUDA is available. Using GPU.")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
elif torch.backends.mps.is_available():
    print("MPS is available. Using Apple Metal.")
else:
    print("Neither CUDA nor MPS is available. Using CPU.")
