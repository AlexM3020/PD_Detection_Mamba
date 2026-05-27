import torch
import torch.nn as nn


# Define the expected class
class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # PyTorch will rebuild from saved data
        pass


# List all PyTorch classes that might be in the model
pytorch_classes = [
    LSTMClassifier,
    nn.LSTM,
    nn.Linear,
    nn.Sequential,
    nn.ReLU,
    nn.Dropout,
    # Add any other nn modules that might be used
]

# Load the model
print("Attempting to load model...")
try:
    with torch.serialization.safe_globals(pytorch_classes):
        model = torch.load('lstm_parkinson_full_2.pt',
                           map_location='cpu',
                           weights_only=True)
    print("Loaded with weights_only=True (safe mode)")
except Exception as e1:
    print(f"Safe loading failed: {e1}")
    print("\nTrying with weights_only=False (unsafe mode)...")
    try:
        model = torch.load('lstm_parkinson_full_2.pt',
                           map_location='cpu',
                           weights_only=False)
        print("Loaded with weights_only=False (unsafe mode)")
    except Exception as e2:
        print(f"Failed to load model: {e2}")
        exit(1)

# ============================================================================
# INSPECT THE MODEL
# ============================================================================
print("\n" + "=" * 60)
print("MODEL INSPECTION")
print("=" * 60)

print(f"\nModel type: {type(model)}")
print(f"Model class name: {model.__class__.__name__}")

# Try to print the model structure
print("\n" + "-" * 60)
print("MODEL ARCHITECTURE")
print("-" * 60)
try:
    print(model)
except:
    print("Could not print full model structure")

# Count parameters
print("\n" + "-" * 60)
print("PARAMETER COUNT")
print("-" * 60)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# List all layers and their shapes
print("\n" + "-" * 60)
print("LAYER DETAILS")
print("-" * 60)

# Method 1: Using named_modules (most detailed)
for name, module in model.named_modules():
    if name != '':  # Skip the root module
        print(f"\n{name}:")
        print(f"  Type: {type(module).__name__}")

        # Get parameter shapes
        param_count = 0
        for param_name, param in module.named_parameters(recurse=False):
            print(f"  {param_name}: {tuple(param.shape)}")
            param_count += param.numel()

        if param_count > 0:
            print(f"  Total parameters in layer: {param_count:,}")

        # Get specific attributes for common layer types
        if isinstance(module, nn.LSTM):
            print(f"  input_size: {module.input_size}")
            print(f"  hidden_size: {module.hidden_size}")
            print(f"  num_layers: {module.num_layers}")
            print(f"  bidirectional: {module.bidirectional}")
            print(f"  batch_first: {module.batch_first}")
        elif isinstance(module, nn.Linear):
            print(f"  in_features: {module.in_features}")
            print(f"  out_features: {module.out_features}")
        elif isinstance(module, nn.Dropout):
            print(f"  p: {module.p}")

# Method 2: Just list all parameters
print("\n" + "-" * 60)
print("ALL PARAMETERS (by name)")
print("-" * 60)
for name, param in model.named_parameters():
    print(f"{name}: {tuple(param.shape)}")

# Check if model has forward method
print("\n" + "-" * 60)
print("MODEL CAPABILITIES")
print("-" * 60)
print(f"Has forward method: {hasattr(model, 'forward')}")
print(f"Has __call__ method: {hasattr(model, '__call__')}")

# Try a simple forward pass to verify model works
print("\n" + "-" * 60)
print("MODEL VERIFICATION")
print("-" * 60)
try:
    # Create dummy input (adjust shape based on your expected input)
    # Common shapes: (batch_size, sequence_length, input_features)
    dummy_input = torch.randn(1, 100, 10)  # Adjust based on your data

    print(f"Dummy input shape: {tuple(dummy_input.shape)}")

    with torch.no_grad():
        output = model(dummy_input)
        print(f"Output shape: {tuple(output.shape)}")
        print(f"Model forward pass successful!")
except Exception as e:
    print(f"Forward pass failed: {e}")
    print("You may need to adjust the dummy input shape")

print("\n" + "=" * 60)
print("INSPECTION COMPLETE")
print("=" * 60)