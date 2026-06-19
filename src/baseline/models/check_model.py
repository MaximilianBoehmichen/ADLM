import torch
from src.baseline.models.resnet8_imagenet import ImageNetResNet8

def test_model_forward():
    
    model = resnet8_imagenet(num_classes=10)
    
    
    dummy_input = torch.randn(2, 3, 224, 224)
    
    
    try:
        output = model(dummy_input)
        print("Forward pass successful!")
        print(f"output size: {output.shape}")  # should be torch.Size([2, 10])
    except Exception as e:
        print(f"error: {e}")

if __name__ == "__main__":
    test_model_forward()