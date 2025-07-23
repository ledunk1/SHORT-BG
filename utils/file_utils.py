import os
from pathlib import Path
from typing import List, Tuple

# Supported file extensions
SUPPORTED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
SUPPORTED_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
SUPPORTED_EXTENSIONS = SUPPORTED_VIDEO_EXTENSIONS | SUPPORTED_IMAGE_EXTENSIONS

def is_supported_file(file_path: str) -> bool:
    """
    Check if file is supported
    
    Args:
        file_path: Path to file
        
    Returns:
        True if file is supported, False otherwise
    """
    return Path(file_path).suffix.lower() in SUPPORTED_EXTENSIONS

def get_supported_files(folder_path: str) -> List[str]:
    """
    Get list of supported files in folder
    
    Args:
        folder_path: Path to folder
        
    Returns:
        List of supported file paths
    """
    if not os.path.exists(folder_path):
        return []
    
    supported_files = []
    
    try:
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            
            # Skip directories
            if os.path.isdir(file_path):
                continue
            
            # Check if file is supported
            if is_supported_file(file_path):
                supported_files.append(file_path)
        
        # Sort files alphabetically
        supported_files.sort()
        
    except PermissionError:
        print(f"Permission denied accessing folder: {folder_path}")
    except Exception as e:
        print(f"Error reading folder {folder_path}: {e}")
    
    return supported_files

def get_file_info(file_path: str) -> dict:
    """
    Get file information
    
    Args:
        file_path: Path to file
        
    Returns:
        Dictionary with file information
    """
    if not os.path.exists(file_path):
        return {}
    
    try:
        stat = os.stat(file_path)
        path_obj = Path(file_path)
        
        return {
            'name': path_obj.name,
            'stem': path_obj.stem,
            'suffix': path_obj.suffix,
            'size': stat.st_size,
            'size_mb': round(stat.st_size / (1024 * 1024), 2),
            'modified': stat.st_mtime,
            'is_video': path_obj.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS,
            'is_image': path_obj.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        }
    except Exception as e:
        print(f"Error getting file info for {file_path}: {e}")
        return {}

def create_output_filename(input_filename: str, suffix: str = "", extension: str = ".mp4") -> str:
    """
    Create output filename based on input filename
    
    Args:
        input_filename: Input filename
        suffix: Suffix to add
        extension: Output file extension
        
    Returns:
        Output filename
    """
    path_obj = Path(input_filename)
    stem = path_obj.stem
    
    if suffix:
        return f"{stem}_{suffix}{extension}"
    else:
        return f"{stem}_output{extension}"

def ensure_directory_exists(directory_path: str) -> bool:
    """
    Ensure directory exists, create if it doesn't
    
    Args:
        directory_path: Path to directory
        
    Returns:
        True if directory exists or was created, False otherwise
    """
    try:
        os.makedirs(directory_path, exist_ok=True)
        return True
    except Exception as e:
        print(f"Error creating directory {directory_path}: {e}")
        return False

def get_unique_filename(file_path: str) -> str:
    """
    Get unique filename by adding number suffix if file exists
    
    Args:
        file_path: Desired file path
        
    Returns:
        Unique file path
    """
    if not os.path.exists(file_path):
        return file_path
    
    path_obj = Path(file_path)
    base_name = path_obj.stem
    extension = path_obj.suffix
    directory = path_obj.parent
    
    counter = 1
    while True:
        new_name = f"{base_name}_{counter}{extension}"
        new_path = directory / new_name
        
        if not os.path.exists(new_path):
            return str(new_path)
        
        counter += 1
        if counter > 1000:  # Safety limit
            break
    
    return file_path

def match_files_by_name(primary_files: List[str], secondary_files: List[str]) -> List[Tuple[str, str]]:
    """
    Match files by their base names (without extension)
    
    Args:
        primary_files: List of primary file paths
        secondary_files: List of secondary file paths
        
    Returns:
        List of tuples (primary_file, secondary_file) for matched pairs
    """
    matched_pairs = []
    
    # Create mapping of base names to secondary files
    secondary_map = {}
    for secondary_file in secondary_files:
        base_name = Path(secondary_file).stem.lower()  # Case insensitive
        secondary_map[base_name] = secondary_file
    
    # Match primary files with secondary files
    for primary_file in primary_files:
        primary_base = Path(primary_file).stem.lower()  # Case insensitive
        
        if primary_base in secondary_map:
            secondary_file = secondary_map[primary_base]
            matched_pairs.append((primary_file, secondary_file))
        else:
            # No matching secondary file found
            print(f"⚠️ No matching secondary file for: {Path(primary_file).name}")
    
    return matched_pairs

def get_matched_file_pairs(primary_folder: str, secondary_folder: str) -> List[Tuple[str, str]]:
    """
    Get matched file pairs from two folders based on filename
    
    Args:
        primary_folder: Path to primary folder
        secondary_folder: Path to secondary folder
        
    Returns:
        List of tuples (primary_file, secondary_file) for matched pairs
    """
    if not os.path.exists(primary_folder):
        print(f"❌ Primary folder does not exist: {primary_folder}")
        return []
    
    if not os.path.exists(secondary_folder):
        print(f"❌ Secondary folder does not exist: {secondary_folder}")
        return []
    
    # Get supported files from both folders
    primary_files = get_supported_files(primary_folder)
    secondary_files = get_supported_files(secondary_folder)
    
    if not primary_files:
        print(f"❌ No supported files found in primary folder: {primary_folder}")
        return []
    
    if not secondary_files:
        print(f"❌ No supported files found in secondary folder: {secondary_folder}")
        return []
    
    # Match files by name
    matched_pairs = match_files_by_name(primary_files, secondary_files)
    
    print(f"\n📊 File Matching Results:")
    print(f"  📁 Primary files: {len(primary_files)}")
    print(f"  📁 Secondary files: {len(secondary_files)}")
    print(f"  ✅ Matched pairs: {len(matched_pairs)}")
    
    if matched_pairs:
        print(f"\n🔗 Matched File Pairs:")
        for i, (primary, secondary) in enumerate(matched_pairs, 1):
            print(f"  {i}. {Path(primary).name} ↔ {Path(secondary).name}")
    
    unmatched_primary = len(primary_files) - len(matched_pairs)
    if unmatched_primary > 0:
        print(f"\n⚠️ Unmatched primary files: {unmatched_primary}")
        matched_names = {Path(p).stem.lower() for p, s in matched_pairs}
        for primary_file in primary_files:
            if Path(primary_file).stem.lower() not in matched_names:
                print(f"    📄 {Path(primary_file).name}")
    
    return matched_pairs

def validate_file_pairs(file_pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    Validate that all file pairs exist and are accessible
    
    Args:
        file_pairs: List of file pairs to validate
        
    Returns:
        List of valid file pairs
    """
    valid_pairs = []
    
    for primary_file, secondary_file in file_pairs:
        primary_valid = os.path.exists(primary_file) and os.access(primary_file, os.R_OK)
        secondary_valid = os.path.exists(secondary_file) and os.access(secondary_file, os.R_OK)
        
        if primary_valid and secondary_valid:
            valid_pairs.append((primary_file, secondary_file))
        else:
            if not primary_valid:
                print(f"❌ Primary file not accessible: {Path(primary_file).name}")
            if not secondary_valid:
                print(f"❌ Secondary file not accessible: {Path(secondary_file).name}")
    
    return valid_pairs