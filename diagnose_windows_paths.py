import os
import h5py
from pathlib import Path

def diagnose():
    target = r'C:\Users\CIBMn\Desktop\abhishek\comparison\gedai_official\gedai\data\fsavLEADFIELD_4_GEDAI.mat'
    print(f"--- DIAGNOSING PATH: {target} ---")
    
    # 1. OS check
    p = Path(target)
    print(f"Path exists (Pathlib): {p.exists()}")
    print(f"Path exists (os.path): {os.path.exists(target)}")
    
    if not p.exists():
        print("!! Checking parent directories...")
        curr = p
        while curr.parent != curr:
            curr = curr.parent
            print(f"Checking: {curr} -> {'EXISTS' if curr.exists() else 'MISSING'}")
        return

    # 2. File Stat
    print(f"File size: {p.stat().st_size} bytes")
    
    # 3. Basic Read Test
    try:
        with open(target, 'rb') as f:
            chunk = f.read(100)
            print(f"Successfully read first 100 bytes of file.")
    except Exception as e:
        print(f"!! Failed basic open(): {e}")

    # 4. H5PY Test
    print(f"Attempting h5py.File(target, 'r')...")
    try:
        with h5py.File(target, 'r') as f:
            print("Successfully opened with h5py!")
            print(f"Keys in file: {list(f.keys())}")
    except Exception as e:
        print(f"!! H5PY FAILURE: {e}")
        
    # 5. Reverse Path Check
    print("\n--- Listing directory contents of gedai_official/gedai/data ---")
    data_dir = p.parent
    if data_dir.exists():
        for item in data_dir.iterdir():
            print(f" - {item.name}")
    else:
        print("!! data folder not found")

if __name__ == "__main__":
    diagnose()
