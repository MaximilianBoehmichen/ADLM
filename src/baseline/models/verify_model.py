import os
import sys
import torch

# 1. Get the absolute path of the current script (.../src/baseline/models)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. CRUCIAL: Only add 'src' to sys.path. Do NOT add the ADLM root.
# This forces Python to treat 'baseline' as the top-level package, eliminating the loop.
src_dir = os.path.abspath(os.path.join(current_dir, "../.."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# 3. Clean import WITHOUT 'src.' prefix to perfectly match Max's project style
try:
    from baseline.models.image_net_styled_resnet8 import ImageNetResNet8 as TargetModel
    print("🎯 Successfully detected architecture file: image_net_styled_resnet8.py")
except ModuleNotFoundError:
    from baseline.models.resnet8 import ResNet8 as TargetModel
    print("🎯 Fallback detected architecture file: resnet8.py")

def verify():
    print("🚀 Starting forward pass test with ImageNet input size (224x224)...")
    try:
        # Initialize the model using the correctly mapped namespace
        model = TargetModel(num_classes=10)
        model.eval()
        
        # Simulate real ImageNet input size
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # Forward pass
        output = model(dummy_input)
        print(f"✅ Verification successful! Input (1, 3, 224, 224) passed through the network.")
        print(f"📊 Final output shape (Batch, Classes): {output.shape}")
    except Exception as e:
        print(f"❌ An error occurred during model execution: {e}")

if __name__ == "__main__":
    verify()