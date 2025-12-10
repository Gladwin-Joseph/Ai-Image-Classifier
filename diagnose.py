import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path

print("="*70)
print("DIAGNOSTIC TEST")
print("="*70)

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n1. Device: {DEVICE}")

# Check data folders
print("\n2. Checking data folders...")
data_path = Path("./data")
if data_path.exists():
    folders = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
    print(f"   Folders found: {folders}")
    print(f"   Class 0 = '{folders[0]}' (first alphabetically)")
    print(f"   Class 1 = '{folders[1]}' (second alphabetically)")
    
    # Count images
    for i, folder in enumerate(folders):
        count = len(list((data_path / folder).glob('*')))
        print(f"   {folder}: {count} images")
else:
    print("./data folder not found!")

# Check model files
print("\n3. Checking model files...")
model_dir = Path("./models_phase2")
if model_dir.exists():
    models = list(model_dir.glob("*.pth"))
    for m in models:
        size = m.stat().st_size / 1024 / 1024
        print(f"   {m.name}: {size:.2f} MB")
else:
    print("./models_phase2 folder not found!")

# Define FastCNN (from training)
class FastCNN(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.3):
        super(FastCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(32, 128)
        self.relu = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.fc3 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Try loading model
print("\n4. Loading model...")
model_path = "./models_phase2/best_custom_cnn_fast.pth"
if not os.path.exists(model_path):
    model_path = "./models_phase2/best_custom_cnn.pth"
    print(f"   Trying alternate: {model_path}")

if os.path.exists(model_path):
    try:
        model = FastCNN(num_classes=2, dropout_rate=0.3)
        state = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state)
        model.to(DEVICE)
        model.eval()
        print(f"   ✓ Model loaded: {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Model architecture might not match!")
        exit()
else:
    print(f"No model found!")
    exit()

# Transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Test on TRAINING DATA images
print("\n5. Testing on TRAINING DATA images...")
print("-"*70)

data_path = Path("./data")
folders = sorted([d.name for d in data_path.iterdir() if d.is_dir()])

for class_idx, folder in enumerate(folders):
    folder_path = data_path / folder
    images = list(folder_path.glob('*.jpg'))[:3] + list(folder_path.glob('*.png'))[:3]
    
    if not images:
        print(f"\n   No images in {folder}")
        continue
    
    print(f"\n   Testing images from '{folder}' (should be class {class_idx}):")
    
    for img_path in images[:3]:  # Test 3 images
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                output = model(img_tensor)
                probs = torch.softmax(output, dim=1)
                pred = output.argmax(1).item()
            
            status = "✓" if pred == class_idx else "✗ WRONG!"
            print(f"      {img_path.name[:30]:30} -> Pred: {pred}, Probs: [{probs[0][0]:.2%}, {probs[0][1]:.2%}] {status}")
        except Exception as e:
            print(f"      Error: {e}")

print("\n" + "="*70)
print("INTERPRETATION:")
print("="*70)
print("""
If images from 'ai_generated' folder predict as class 0 -> CORRECT
If images from 'real' folder predict as class 1 -> CORRECT

If ALL images predict the SAME class -> Model is broken/biased

Check:
- Was training data balanced? (50% real, 50% AI)
- Did training complete properly?
- Check training logs for final accuracy
""")