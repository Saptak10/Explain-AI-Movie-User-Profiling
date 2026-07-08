import csv
import numpy as np

def convert_csv_to_mmap(csv_path: str, npy_path: str):
    print(f"Scanning {csv_path} to count rows...")
    with open(csv_path, 'r', encoding='utf-8') as f:
        # Subtract 1 for the header
        num_rows = sum(1 for _ in f) - 1
        
    print(f"Allocating memory-mapped binary file for {num_rows:,} rows...")
    
    # Define a structured data type for exact memory efficiency
    dtype = np.dtype([
        ('userId', np.int32), 
        ('movieId', np.int32), 
        ('rating', np.float32)
    ])
    
    # Open the file in write mode; this allocates the space on disk
    mmap_arr = np.lib.format.open_memmap(
        npy_path, mode='w+', dtype=dtype, shape=(num_rows,)
    )
    
    print("Parsing CSV and writing binary data...")
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            mmap_arr[i] = (
                int(row['userId']), 
                int(row['movieId']), 
                float(row['rating'])
            )
            
            if i > 0 and i % 5_000_000 == 0:
                print(f"  ...processed {i:,} rows")
                
    # Flush changes from RAM to disk
    mmap_arr.flush()
    print(f"\nSuccess! Binary dataset saved to {npy_path}")

if __name__ == "__main__":
    # Adjust these paths if necessary
    convert_csv_to_mmap(
        csv_path="app/database/ml-latest/ratings.csv",
        npy_path="app/database/ml-latest/ratings.npy"
    )