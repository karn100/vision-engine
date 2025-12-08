from core.layers.conv import conv2d
import torch

if __name__ == "__main__":
    conv = conv2d(3,8,kernel_size=3,padding=1)
    x = torch.randn(1,3,32,32)
    y = conv(x)
    print(y.shape)

    y.mean().backward()
    print("Backward OK")