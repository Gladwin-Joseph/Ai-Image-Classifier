import os
from pathlib import Path

def check_dataset_structure(data_dir):
    """Check if dataset has correct structure for ImageFolder"""
    
    print(f"\n{'='*70}")
    print("DATASET STRUCTURE CHECKER")
    print(f"{'='*70}\n")
    
    data_path = Path(data_dir)
    
    # Check if directory exists
    if not data_path.exists():
        print(f"ERROR: Directory '{data_dir}' does not exist!")
        print(f"\nPlease create the directory and add your images.")
        return False
    
    print(f"✓ Directory exists: {data_dir}")
    
    # List all contents
    contents = list(data_path.iterdir())
    print(f"\nContents of {data_dir}:")
    for item in contents:
        print(f"   {'[DIR]' if item.is_dir() else '[FILE]'} {item.name}")
    
    # Check for class folders
    class_folders = [d for d in contents if d.is_dir()]
    
    if len(class_folders) == 0:
        print(f"\nERROR: No class folders found!")
        print(f"\nExpected structure:")
        print(f"   {data_dir}/")
        print(f"   ├── ai_generated/")
        print(f"   │   ├── image1.jpg")
        print(f"   │   ├── image2.png")
        print(f"   │   └── ...")
        print(f"   └── real/")
        print(f"       ├── image1.jpg")
        print(f"       ├── image2.png")
        print(f"       └── ...")
        return False
    
    print(f"\n✓ Found {len(class_folders)} class folder(s):")
    
    # Supported image extensions
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff', '.ppm', '.pgm')
    
    total_images = 0
    valid_structure = True
    
    for class_folder in class_folders:
        class_name = class_folder.name
        
        # Get all image files
        image_files = [
            f for f in class_folder.iterdir() 
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        
        num_images = len(image_files)
        total_images += num_images
        
        print(f"\n{class_name}/")
        print(f"      Images: {num_images}")
        
        if num_images == 0:
            print(f"WARNING: No valid images found in this folder!")
            valid_structure = False
        else:
            # Show first 5 files
            print(f"      Sample files:")
            for img in image_files[:5]:
                print(f"         - {img.name}")
            if num_images > 5:
                print(f"         ... and {num_images - 5} more")
    
    print(f"\n{'='*70}")
    if total_images == 0:
        print("RESULT: No valid images found in any class folder!")
        print(f"\nSupported image formats: {', '.join(image_extensions)}")
        valid_structure = False
    elif not valid_structure:
        print("RESULT: Dataset structure has issues (see warnings above)")
    else:
        print("RESULT: Dataset structure is VALID!")
        print(f"\nSummary:")
        print(f"   Total classes: {len(class_folders)}")
        print(f"   Total images: {total_images}")
        print(f"   Average images per class: {total_images / len(class_folders):.1f}")
    
    print(f"{'='*70}\n")
    
    return valid_structure

def suggest_fix(data_dir):
    """Suggest how to fix common issues"""
    
    data_path = Path(data_dir)
    
    print(f"\n{'='*70}")
    print("SUGGESTED FIXES")
    print(f"{'='*70}\n")
    
    # Check if images are directly in data_dir
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff', '.ppm', '.pgm')
    direct_images = [
        f for f in data_path.iterdir() 
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    if direct_images:
        print("Found images directly in the data directory!")
        print(f"   Number of images: {len(direct_images)}")
        print(f"\nThese images should be organized into class folders.")
        print(f"\n   Current structure:")
        print(f"   {data_dir}/")
        print(f"   ├── image1.jpg")
        print(f"   ├── image2.png")
        print(f"   └── ...")
        print(f"\n   Required structure:")
        print(f"   {data_dir}/")
        print(f"   ├── ai_generated/")
        print(f"   │   ├── image1.jpg  ✓")
        print(f"   │   └── image2.png  ✓")
        print(f"   └── real/")
        print(f"       ├── image3.jpg  ✓")
        print(f"       └── image4.png  ✓")
        
        print(f"\nTo fix manually:")
        print(f"   1. Create folders: {data_dir}/ai_generated and {data_dir}/real")
        print(f"   2. Move AI-generated images to {data_dir}/ai_generated/")
        print(f"   3. Move real images to {data_dir}/real/")
    
    # Check if there are unexpected files in class folders
    class_folders = [d for d in data_path.iterdir() if d.is_dir()]
    for class_folder in class_folders:
        non_image_files = [
            f for f in class_folder.iterdir()
            if f.is_file() and f.suffix.lower() not in image_extensions
        ]
        
        if non_image_files:
            print(f"\nFound non-image files in {class_folder.name}/:")
            for f in non_image_files[:10]:
                print(f"   - {f.name}")
            if len(non_image_files) > 10:
                print(f"   ... and {len(non_image_files) - 10} more")
            print(f"\nThese files will be ignored, but consider removing them.")
    
    print(f"\n{'='*70}\n")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Check dataset structure')
    parser.add_argument('--data_dir', default='./data', help='Data directory')
    args = parser.parse_args()
    
    is_valid = check_dataset_structure(args.data_dir)
    
    if not is_valid:
        suggest_fix(args.data_dir)
        print("\nPlease fix the dataset structure and run this script again.")
        print(f"   Command: python check_data.py --data_dir {args.data_dir}\n")
    else:
        print("\nYour dataset is ready to use!")
        print(f"   You can now run: python train.py --data_dir {args.data_dir}\n")