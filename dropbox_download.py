"""
Bulk download images/ folder from Dropbox, preserving subfolder structure.

Setup:
1. Create a Dropbox app at https://www.dropbox.com/developers/apps
   - Choose "Scoped access" and "Full Dropbox" (or App folder if images/ is there)
   - Under Permissions, enable: files.metadata.read, files.content.read
2. Generate an access token (or use refresh token for long-running jobs)
3. Install SDK: pip install dropbox

Usage:
    python dropbox_download.py 
    python dropbox_download.py --dropbox-path /images --local-path ./downloaded_images
"""

import argparse
import dropbox
from dropbox.exceptions import ApiError, AuthError
from dropbox.files import FileMetadata
from pathlib import Path
import time
import sys
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import DROPBOX_TOKEN

# Rate limiting and retry settings
MAX_RETRIES = 5
BASE_DELAY = 1.0
MAX_WORKERS = 12  # parallel downloads


def create_client(token):
    """Create Dropbox client and verify connection."""
    try:
        dbx = dropbox.Dropbox(token)
        dbx.users_get_current_account()
        print("Successfully connected to Dropbox")
        return dbx
    except AuthError:
        print("ERROR: Invalid access token. Please check your token.")
        sys.exit(1)


def list_all_files(dbx, folder_path):
    """Recursively list all files in folder, handling pagination."""
    files = []
    
    try:
        result = dbx.files_list_folder(folder_path, recursive=True)
    except ApiError as e:
        if e.error.is_path() and e.error.get_path().is_not_found():
            print(f"ERROR: Folder '{folder_path}' not found in Dropbox")
            sys.exit(1)
        raise
    
    while True:
        for entry in result.entries:
            if isinstance(entry, FileMetadata):
                files.append(entry)
        
        if not result.has_more:
            break
        
        result = dbx.files_list_folder_continue(result.cursor)
        print(f"  Listing files... {len(files)} found", end='\r')
    
    print(f"  Found {len(files)} files total          ")
    return files


def download_file(dbx, file_meta, dropbox_root, local_root, retries=MAX_RETRIES):
    """Download single file with retry logic."""
    # Get the root folder name (e.g., "images" from "/images")
    root_folder_name = dropbox_root.rstrip('/').split('/')[-1]
    
    # Compute relative path, keeping the root folder name
    path_after_root = file_meta.path_display[len(dropbox_root):].lstrip('/')
    rel_path = f"{root_folder_name}/{path_after_root}"
    local_path = local_root / rel_path
    
    # Skip if already exists with same size
    if local_path.exists() and local_path.stat().st_size == file_meta.size:
        return {"status": "skipped", "path": rel_path}
    
    # Create parent directories
    local_path.parent.mkdir(parents=True, exist_ok=True)
    
    for attempt in range(retries):
        try:
            dbx.files_download_to_file(str(local_path), file_meta.path_lower)
            return {"status": "downloaded", "path": rel_path, "size": file_meta.size}
        
        except ApiError as e:
            error_msg = f"ApiError: {e.error if hasattr(e, 'error') else str(e)}"
            if attempt < retries - 1:
                delay = BASE_DELAY * (2 ** attempt)
                time.sleep(delay)
            else:
                return {"status": "failed", "path": rel_path, "error": error_msg}
        
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            if attempt < retries - 1:
                delay = BASE_DELAY * (2 ** attempt)
                time.sleep(delay)
            else:
                return {"status": "failed", "path": rel_path, "error": error_msg}
    
    return {"status": "failed", "path": rel_path, "error": "max retries exceeded"}


def save_progress(progress_file, completed, failed):
    """Save progress for resumability."""
    with open(progress_file, 'w') as f:
        json.dump({"completed": list(completed), "failed": list(failed)}, f)


def load_progress(progress_file):
    """Load previous progress if exists."""
    if progress_file.exists():
        with open(progress_file) as f:
            data = json.load(f)
            return set(data.get("completed", [])), set(data.get("failed", []))
    return set(), set()


def main():
    parser = argparse.ArgumentParser(description="Bulk download folder from Dropbox")
    parser.add_argument("--dropbox-path", default="italaw_data/images", 
                        help="Path in Dropbox to download (default: /images)")
    parser.add_argument("--local-path", default="D:/",
                        help="Local destination path (default: D:/)")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS,
                        help=f"Parallel download threads (default: {MAX_WORKERS})")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previous progress")
    parser.add_argument("--extensions", default=".png",
                        help="File extensions to download (default: .png)")
    args = parser.parse_args()
    
    # Normalize paths
    dropbox_path = args.dropbox_path.rstrip('/')
    if not dropbox_path.startswith('/'):
        dropbox_path = '/' + dropbox_path
    
    local_root = Path(args.local_path)
    local_root.mkdir(parents=True, exist_ok=True)
    
    progress_file = local_root / ".download_progress.json"
    valid_extensions = set(ext.strip().lower() for ext in args.extensions.split(','))
    
    print(f"Dropbox folder: {dropbox_path}")
    print(f"Local destination: {local_root.absolute()}")
    print(f"File types: {valid_extensions}")
    print()
    
    # Connect
    dbx = create_client(DROPBOX_TOKEN)
    
    # List all files
    print("Listing files in Dropbox...")
    all_files = list_all_files(dbx, dropbox_path)
    
    # Filter to images only
    image_files = [
        f for f in all_files 
        if Path(f.name).suffix.lower() in valid_extensions
    ]
    print(f"Filtered to {len(image_files)} image files")
    
    # Load progress if resuming
    completed, failed = set(), set()
    if args.resume:
        completed, failed = load_progress(progress_file)
        print(f"Resuming: {len(completed)} already completed, {len(failed)} previously failed")
    
    # Filter out already completed
    to_download = [f for f in image_files if f.path_lower not in completed]
    print(f"Files to download: {len(to_download)}")
    print()
    
    if not to_download:
        print("Nothing to download!")
        return
    
    # Calculate total size
    total_bytes = sum(f.size for f in to_download)
    print(f"Total download size: {total_bytes / (1024**3):.2f} GB")
    print()
    
    # Download with progress
    downloaded_count = 0
    downloaded_bytes = 0
    failed_files = []
    start_time = time.time()
    
    # Need separate client per thread
    def worker_download(file_meta):
        worker_dbx = dropbox.Dropbox(DROPBOX_TOKEN)
        return download_file(worker_dbx, file_meta, dropbox_path, local_root)
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(worker_download, f): f for f in to_download}
        
        for future in as_completed(futures):
            file_meta = futures[future]
            result = future.result()
            
            if result["status"] == "downloaded":
                downloaded_count += 1
                downloaded_bytes += result["size"]
                completed.add(file_meta.path_lower)
            elif result["status"] == "skipped":
                downloaded_count += 1
                completed.add(file_meta.path_lower)
            else:
                failed_files.append(result)
                failed.add(file_meta.path_lower)
            
            # Progress update
            elapsed = time.time() - start_time
            rate = downloaded_bytes / elapsed if elapsed > 0 else 0
            pct = (downloaded_count + len(failed_files)) / len(to_download) * 100
            
            print(f"Progress: {pct:.1f}% | "
                  f"Downloaded: {downloaded_count} | "
                  f"Failed: {len(failed_files)} | "
                  f"Rate: {rate / (1024**2):.1f} MB/s", end='\r')
            
            # Save progress periodically
            if (downloaded_count + len(failed_files)) % 100 == 0:
                save_progress(progress_file, completed, failed)
    
    # Final save
    save_progress(progress_file, completed, failed)
    
    print()
    print()
    print("=" * 60)
    print("DOWNLOAD COMPLETE")
    print(f"  Successfully downloaded: {downloaded_count}")
    print(f"  Failed: {len(failed_files)}")
    print(f"  Total time: {time.time() - start_time:.1f} seconds")
    print(f"  Files saved to: {local_root.absolute()}")
    
    if failed_files:
        print()
        print("Failed files:")
        for f in failed_files[:10]:
            print(f"  {f['path']}: {f.get('error', 'unknown error')}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more")
        print()
        print("Run with --resume to retry failed files")


if __name__ == "__main__":
    main()