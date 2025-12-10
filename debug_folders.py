import os
from pathlib import Path

def debug_folders():
    print("\nFOLDER DEBUG\n")
    
    current_dir = Path(".")
    
    # List all folders
    print("All folders in current directory:")
    for item in current_dir.iterdir():
        if item.is_dir():
            print(f"{item.name}/")
    
    print("\n" + "="*50)
    
    # Check train folder
    train_path = Path("train")
    if train_path.exists():
        images = list(train_path.glob("*.*"))
        print(f"\nTrain folder: {len(images)} images")
        if len(images) > 0:
            print("  First 5 files:")
            for img in images[:5]:
                print(f"    - {img.name}")
        else:
            print("NO IMAGES FOUND!")
            print("Contents:")
            for item in train_path.iterdir():
                print(f"    - {item.name} ({'DIR' if item.is_dir() else 'FILE'})")
    else:
        print("\ntrain folder NOT FOUND")
    
    print("\n" + "="*50)
    
    # Check test folder
    test_path = Path("test")
    if test_path.exists():
        images = list(test_path.glob("*.*"))
        print(f"\nTest folder: {len(images)} images")
        if len(images) > 0:
            print("  First 5 files:")
            for img in images[:5]:
                print(f"    - {img.name}")
        else:
            print("NO IMAGES FOUND!")
            print("  Contents:")
            for item in test_path.iterdir():
                print(f"    - {item.name} ({'DIR' if item.is_dir() else 'FILE'})")
    else:
        print("\ntest folder NOT FOUND")
    
    print("\n" + "="*50)
    print("\nSUMMARY:")
    print("If you see 0 images in train/test folders:")
    print("1. Your CIFAKE files might be in a subfolder")
    print("2. Try renaming your extracted folders to 'train' and 'test'")
    print("3. Make sure images are directly in those folders (not in subfolders)")

if __name__ == "__main__":
    debug_folders()