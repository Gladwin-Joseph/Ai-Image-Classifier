import os
import shutil
from pathlib import Path
from tqdm import tqdm

def reorganize_kaggle_dataset(source_dir, dest_dir="./data"):
    """
    Reorganize Kaggle dataset from:
      source/
        train/
          REAL/
          FAKE/
        test/
          REAL/
          FAKE/
    
    To:
      data/
        ai_generated/  (all FAKE images)
        real/          (all REAL images)
    """
    
    print("\n" + "="*70)
    print("KAGGLE DATASET REORGANIZER")
    print("="*70)
    print("\nConverting structure from:")
    print("  train/REAL, train/FAKE")
    print("  test/REAL, test/FAKE")
    print("\nTo:")
    print("  data/ai_generated/")
    print("  data/real/")
    print("="*70 + "\n")
    
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    
    # Verify source exists
    if not source_path.exists():
        print(f"ERROR: Source directory not found: {source_dir}")
        print("\nPlease provide the path to your Kaggle dataset folder")
        return False
    
    print(f"✓ Source directory found: {source_path.absolute()}\n")
    
    # Check for train/test folders
    train_dir = source_path / "train"
    test_dir = source_path / "test"
    
    has_train = train_dir.exists()
    has_test = test_dir.exists()
    
    print("Checking source structure:")
    print(f"  {'✓' if has_train else '✗'} train/ folder")
    print(f"  {'✓' if has_test else '✗'} test/ folder")
    
    if not (has_train or has_test):
        print("\nERROR: No train/ or test/ folders found!")
        print(f"\nCurrent contents of {source_dir}:")
        for item in source_path.iterdir():
            print(f"   {'[DIR]' if item.is_dir() else '[FILE]'} {item.name}")
        return False
    
    print()
    
    # Create destination directories
    ai_dir = dest_path / "ai_generated"
    real_dir = dest_path / "real"
    
    # Remove old destination if exists
    if dest_path.exists():
        print(f"Destination folder already exists: {dest_path}")
        response = input("Delete existing data folder and create new one? (yes/no): ").strip().lower()
        if response == "yes":
            print("Removing old data folder...")
            shutil.rmtree(dest_path)
        else:
            print("Operation cancelled.")
            return False
    
    # Create new directories
    ai_dir.mkdir(parents=True, exist_ok=True)
    real_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n✓ Created destination folders in: {dest_path.absolute()}\n")
    
    # Image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff', '.ppm', '.pgm'}
    
    total_ai = 0
    total_real = 0
    
    # Process train and test folders
    for subset in ["train", "test"]:
        subset_dir = source_path / subset
        
        if not subset_dir.exists():
            continue
        
        print(f"{'='*70}")
        print(f"Processing {subset.upper()}/ folder")
        print(f"{'='*70}")
        
        # Look for FAKE folder (various possible names)
        fake_folder = None
        for name in ["FAKE", "fake", "Fake", "AI", "ai", "ai_generated"]:
            test_path = subset_dir / name
            if test_path.exists():
                fake_folder = test_path
                print(f"✓ Found FAKE images in: {name}/")
                break
        
        if fake_folder:
            # Get all image files
            images = []
            for ext in image_extensions:
                images.extend(list(fake_folder.glob(f"*{ext}")))
                images.extend(list(fake_folder.glob(f"*{ext.upper()}")))
            
            print(f"  Found {len(images)} FAKE images")
            
            # Copy to ai_generated with prefix
            copied = 0
            for img in tqdm(images, desc=f"  Copying to ai_generated"):
                new_name = f"{subset}_{img.name}"
                dest = ai_dir / new_name
                
                # Handle duplicates
                counter = 1
                while dest.exists():
                    stem = img.stem
                    suffix = img.suffix
                    new_name = f"{subset}_{stem}_{counter}{suffix}"
                    dest = ai_dir / new_name
                    counter += 1
                
                shutil.copy2(img, dest)
                copied += 1
            
            total_ai += copied
            print(f"  ✓ Copied {copied} images to ai_generated/\n")
        else:
            print(f"No FAKE folder found in {subset}/\n")
        
        # Look for REAL folder
        real_folder = None
        for name in ["REAL", "real", "Real"]:
            test_path = subset_dir / name
            if test_path.exists():
                real_folder = test_path
                print(f"✓ Found REAL images in: {name}/")
                break
        
        if real_folder:
            # Get all image files
            images = []
            for ext in image_extensions:
                images.extend(list(real_folder.glob(f"*{ext}")))
                images.extend(list(real_folder.glob(f"*{ext.upper()}")))
            
            print(f"  Found {len(images)} REAL images")
            
            # Copy to real with prefix
            copied = 0
            for img in tqdm(images, desc=f"  Copying to real"):
                new_name = f"{subset}_{img.name}"
                dest = real_dir / new_name
                
                # Handle duplicates
                counter = 1
                while dest.exists():
                    stem = img.stem
                    suffix = img.suffix
                    new_name = f"{subset}_{stem}_{counter}{suffix}"
                    dest = real_dir / new_name
                    counter += 1
                
                shutil.copy2(img, dest)
                copied += 1
            
            total_real += copied
            print(f"  ✓ Copied {copied} images to real/\n")
        else:
            print(f"No REAL folder found in {subset}/\n")
    
    # Summary
    print("="*70)
    print("REORGANIZATION COMPLETE!")
    print("="*70)
    print(f"\nSUMMARY:")
    print(f"   Total AI-Generated (FAKE): {total_ai} images")
    print(f"   Total Real: {total_real} images")
    print(f"   Grand Total: {total_ai + total_real} images")
    
    print(f"\nNEW STRUCTURE:")
    print(f"   {dest_path.absolute()}/")
    print(f"   ├── ai_generated/ ({total_ai} images)")
    print(f"   └── real/ ({total_real} images)")
    
    print(f"\n{'='*70}")
    print("NEXT STEPS:")
    print(f"{'='*70}")
    print("\n1. Verify the data:")
    print("   python check_data.py --data_dir ./data")
    print("\n2. Train the model:")
    print("   python train.py --data_dir ./data")
    print(f"\n{'='*70}\n")
    
    return True

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Reorganize Kaggle dataset')
    parser.add_argument('--source', required=True, help='Source directory with train/test folders')
    parser.add_argument('--dest', default='./data', help='Destination directory (default: ./data)')
    
    args = parser.parse_args()
    
    try:
        success = reorganize_kaggle_dataset(args.source, args.dest)
        
        if not success:
            print("\nReorganization failed")
            print("\nMake sure your source directory contains:")
            print("   train/REAL/ and train/FAKE/")
            print("   test/REAL/ and test/FAKE/")
    
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()