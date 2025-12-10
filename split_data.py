import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

def split_data(input_dir='./data', output_dir='./data_splits', train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):    
    random.seed(random_seed)
    
    # Verify ratios
    if train_ratio + val_ratio + test_ratio != 1.0:
        print(f"❌ Ratios must sum to 1.0! Got {train_ratio + val_ratio + test_ratio}")
        return
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Check if input directory exists
    if not input_path.exists():
        print(f"Input directory not found: {input_dir}")
        return
    
    # Create output directories
    splits = ['train', 'val', 'test']
    for split in splits:
        for class_name in ['real', 'ai_generated']:
            (output_path / split / class_name).mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("SPLITTING DATA")
    print(f"{'='*70}\n")
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Split ratios: Train={train_ratio*100:.0f}%, Val={val_ratio*100:.0f}%, Test={test_ratio*100:.0f}%\n")
    
    # Process each class
    class_names = ['real', 'ai_generated']
    
    for class_name in class_names:
        class_dir = input_path / class_name
        
        if not class_dir.exists():
            print(f"Class directory not found: {class_dir}")
            continue
        
        # Get all image files
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(class_dir.glob(f'*{ext}'))
            image_files.extend(class_dir.glob(f'*{ext.upper()}'))
        
        image_files = list(set(image_files))  # Remove duplicates
        
        if not image_files:
            print(f"No images found in {class_dir}")
            continue
        
        print(f"\nProcessing {class_name}: {len(image_files)} images")
        
        # Shuffle and split
        random.shuffle(image_files)
        
        train_count = int(len(image_files) * train_ratio)
        val_count = int(len(image_files) * val_ratio)
        
        train_files = image_files[:train_count]
        val_files = image_files[train_count:train_count + val_count]
        test_files = image_files[train_count + val_count:]
        
        # Copy files to splits
        splits_data = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }
        
        for split_name, files in splits_data.items():
            dest_dir = output_path / split_name / class_name
            
            print(f"  {split_name}: {len(files)} images", end=' ')
            
            for img_file in tqdm(files, disable=True):
                try:
                    shutil.copy2(img_file, dest_dir / img_file.name)
                except Exception as e:
                    print(f"Error copying {img_file}: {e}")
            
            print("✓")
    
    # Print summary
    print(f"\n{'='*70}")
    print("SPLIT SUMMARY")
    print(f"{'='*70}\n")
    
    for split in splits:
        real_count = len(list((output_path / split / 'real').glob('*')))
        ai_count = len(list((output_path / split / 'ai_generated').glob('*')))
        total = real_count + ai_count
        
        print(f"{split.upper():10} | Real: {real_count:6} | AI: {ai_count:6} | Total: {total:6} | Percentage: {total/sum([len(list((output_path / s / 'real').glob('*'))) + len(list((output_path / s / 'ai_generated').glob('*'))) for s in splits])*100:.1f}%")
    
    print(f"\n✓ Data split successfully!")
    print(f"✓ Output directory: {output_dir}")
    
    print(f"\n{'='*70}")
    print("TESTING COMMANDS")
    print(f"{'='*70}\n")
    
    print("Now you can test each split separately:\n")
    print("TEST TRAINING SET (70%):")
    print(f"  python test_phase2_custom_cnn.py --mode batch --test_dir {output_dir}/train\n")
    
    print("TEST VALIDATION SET (15%):")
    print(f"  python test_phase2_custom_cnn.py --mode batch --test_dir {output_dir}/val\n")
    
    print("TEST SET (15%):")
    print(f"  python test_phase2_custom_cnn.py --mode batch --test_dir {output_dir}/test\n")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Split data into train/val/test')
    parser.add_argument(
        '--input',
        type=str,
        default='./data',
        help='Input data directory (default: ./data)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./data_splits',
        help='Output directory for splits (default: ./data_splits)'
    )
    parser.add_argument(
        '--train',
        type=float,
        default=0.7,
        help='Training ratio (default: 0.7)'
    )
    parser.add_argument(
        '--val',
        type=float,
        default=0.15,
        help='Validation ratio (default: 0.15)'
    )
    parser.add_argument(
        '--test',
        type=float,
        default=0.15,
        help='Test ratio (default: 0.15)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    args = parser.parse_args()
    
    split_data(
        input_dir=args.input,
        output_dir=args.output,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        random_seed=args.seed
    )