import catan_engine
import torch

def main():
    print(catan_engine.hello())
    x = torch.randn(4, 3)
    print("Torch ok", x)


if __name__ == "__main__":
    main()
