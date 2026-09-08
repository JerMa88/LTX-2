import os
import io
import sys
import time
from pathlib import Path
import py7zr

INSTALLER_PATH = r"C:\Users\jerry\Downloads\cuda_12.8.1_572.61_windows.exe"
TARGET_DIR = Path(r"C:\Users\jerry\cuda_12.8")

class OffsetFile(io.RawIOBase):
    def __init__(self, raw, offset):
        self.raw = raw
        self.offset = offset
        self.raw.seek(0, 2)
        self.size = self.raw.tell() - offset
        self.raw.seek(offset)

    def seekable(self): return True
    def readable(self): return True

    def seek(self, pos, whence=0):
        if whence == 0: target = self.offset + pos
        elif whence == 1: target = self.raw.tell() + pos
        elif whence == 2: target = self.offset + self.size + pos
        self.raw.seek(target)
        return self.raw.tell() - self.offset

    def tell(self):
        return self.raw.tell() - self.offset

    def readinto(self, b):
        return self.raw.readinto(b)


# Mapping of archive prefix -> relative destination inside TARGET_DIR
# We want standard CUDA directory layout:
# bin/
# include/
# lib/x64/
# nvvm/
PATH_MAPPINGS = [
    # nvcc compiler & tools
    ("cuda_nvcc/nvcc/bin/", "bin/"),
    ("cuda_nvcc/nvcc/include/", "include/"),
    ("cuda_nvcc/nvcc/lib/x64/", "lib/x64/"),
    ("cuda_nvcc/nvcc/nvvm/", "nvvm/"),
    
    # cudart headers & libs
    ("cuda_cudart/cudart/include/", "include/"),
    ("cuda_cudart/cudart/lib/x64/", "lib/x64/"),
    ("cuda_cudart/cudart/bin/", "bin/"),

    # cccl (thrust, cub, cuda/std)
    ("cuda_cccl/thrust/include/", "include/"),

    # nvrtc
    ("cuda_nvrtc/nvrtc_dev/include/", "include/"),
    ("cuda_nvrtc/nvrtc_dev/lib/x64/", "lib/x64/"),
    ("cuda_nvrtc/nvrtc/bin/", "bin/"),

    # cublas & cublasLt
    ("libcublas/cublas_dev/include/", "include/"),
    ("libcublas/cublas_dev/lib/x64/", "lib/x64/"),
    ("libcublas/cublas/bin/", "bin/"),

    # nvjitlink & nvfatbin
    ("libnvjitlink/nvjitlink_dev/include/", "include/"),
    ("libnvjitlink/nvjitlink_dev/lib/x64/", "lib/x64/"),
    ("libnvfatbin/nvfatbin_dev/include/", "include/"),
    ("libnvfatbin/nvfatbin_dev/lib/x64/", "lib/x64/"),
]

def main():
    print(f"Opening installer: {INSTALLER_PATH}")
    start_time = time.time()
    
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    
    raw = open(INSTALLER_PATH, "rb")
    wrapped = io.BufferedReader(OffsetFile(raw, 1037424))
    
    with py7zr.SevenZipFile(wrapped) as archive:
        all_names = archive.getnames()
        print(f"Total entries in archive: {len(all_names)}")
        
        # Determine files to extract and their target destination
        extract_map = {}
        for name in all_names:
            for prefix, target_rel in PATH_MAPPINGS:
                if name.startswith(prefix):
                    rel = name[len(prefix):]
                    if rel: # skip root folder entry itself
                        dest = TARGET_DIR / target_rel / rel
                        extract_map[name] = dest
                    break
        
        print(f"Files to extract: {len(extract_map)}")
        
        # py7zr extraction
        # Extract files matching our target list
        matched_archive_names = list(extract_map.keys())
        temp_extract_dir = TARGET_DIR / "_temp_extract"
        temp_extract_dir.mkdir(parents=True, exist_ok=True)
        
        print("Extracting components from 7z archive...")
        archive.extract(path=str(temp_extract_dir), targets=matched_archive_names)
        
        print("Moving files into standard CUDA hierarchy...")
        moved_count = 0
        for src_rel, dest_path in extract_map.items():
            src_path = temp_extract_dir / src_rel
            if src_path.is_file():
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                if dest_path.exists():
                    dest_path.unlink()
                src_path.replace(dest_path)
                moved_count += 1
        
        # Clean up temp dir
        import shutil
        shutil.rmtree(temp_extract_dir, ignore_errors=True)
        
    elapsed = time.time() - start_time
    print(f"Extraction complete in {elapsed:.1f}s. Successfully organized {moved_count} files into {TARGET_DIR}")

if __name__ == "__main__":
    main()
