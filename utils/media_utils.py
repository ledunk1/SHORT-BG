import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Dict, List
import os

class MediaProcessor:
    """Utility class for media processing operations"""
    
    @staticmethod
    def get_media_type(file_path: str) -> str:
        """
        Determine media type from file extension
        
        Args:
            file_path: Path to media file
            
        Returns:
            Media type: 'image', 'video', or 'unknown'
        """
        ext = Path(file_path).suffix.lower()
        
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp', '.gif'}
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
        
        if ext in image_extensions:
            return 'image'
        elif ext in video_extensions:
            return 'video'
        else:
            return 'unknown'
    
    @staticmethod
    def get_media_info(file_path: str) -> Dict:
        """
        Get comprehensive media information
        
        Args:
            file_path: Path to media file
            
        Returns:
            Dictionary with media information
        """
        info = {
            'type': MediaProcessor.get_media_type(file_path),
            'width': 0,
            'height': 0,
            'duration': 0.0,
            'fps': 30.0,
            'frame_count': 1,
            'file_size': 0,
            'exists': os.path.exists(file_path)
        }
        
        if not info['exists']:
            return info
        
        try:
            info['file_size'] = os.path.getsize(file_path)
        except:
            pass
        
        if info['type'] == 'image':
            try:
                # Try with PIL first
                with Image.open(file_path) as img:
                    info['width'], info['height'] = img.size
                    info['duration'] = 0.0  # Static image
                    info['frame_count'] = 1
            except:
                # Fallback to OpenCV
                try:
                    image = cv2.imread(file_path)
                    if image is not None:
                        info['height'], info['width'] = image.shape[:2]
                except:
                    pass
        
        elif info['type'] == 'video':
            try:
                cap = cv2.VideoCapture(file_path)
                if cap.isOpened():
                    info['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    info['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    info['fps'] = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    info['frame_count'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    info['duration'] = info['frame_count'] / info['fps'] if info['fps'] > 0 else 0.0
                cap.release()
            except:
                pass
        
        return info
    
    @staticmethod
    def resize_with_aspect_ratio(image: np.ndarray, target_width: int, target_height: int, 
                                maintain_aspect: bool = True) -> np.ndarray:
        """
        Resize image with optional aspect ratio maintenance
        
        Args:
            image: Input image
            target_width: Target width
            target_height: Target height
            maintain_aspect: Whether to maintain aspect ratio
            
        Returns:
            Resized image
        """
        if not maintain_aspect:
            # Stretch to exact dimensions
            return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        
        height, width = image.shape[:2]
        
        # Calculate aspect ratios
        target_aspect = target_width / target_height
        image_aspect = width / height
        
        if image_aspect > target_aspect:
            # Image is wider, fit to width
            new_width = target_width
            new_height = int(target_width / image_aspect)
        else:
            # Image is taller, fit to height
            new_height = target_height
            new_width = int(target_height * image_aspect)
        
        # Resize image
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Create output with target dimensions (black background)
        output = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        # Center the resized image
        y_offset = (target_height - new_height) // 2
        x_offset = (target_width - new_width) // 2
        
        output[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized
        
        return output
    
    @staticmethod
    def extract_frames(video_path: str, max_frames: int = None) -> List[np.ndarray]:
        """
        Extract frames from video
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract
            
        Returns:
            List of frames
        """
        frames = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frames.append(frame.copy())
                frame_count += 1
                
                if max_frames and frame_count >= max_frames:
                    break
            
            cap.release()
        except Exception as e:
            print(f"Error extracting frames from {video_path}: {e}")
        
        return frames
    
    @staticmethod
    def create_frame_sequence(frames: List[np.ndarray], target_count: int, 
                            method: str = 'loop') -> List[np.ndarray]:
        """
        Create frame sequence with target count
        
        Args:
            frames: Input frames
            target_count: Target number of frames
            method: 'loop', 'sample', or 'extend'
            
        Returns:
            Frame sequence with target count
        """
        if not frames:
            return []
        
        if len(frames) == target_count:
            return frames.copy()
        
        if method == 'loop':
            # Loop/repeat frames
            result = []
            for i in range(target_count):
                frame_index = i % len(frames)
                result.append(frames[frame_index].copy())
            return result
        
        elif method == 'sample':
            # Sample frames evenly
            if len(frames) > target_count:
                # Downsample
                result = []
                step = len(frames) / target_count
                for i in range(target_count):
                    frame_index = int(i * step)
                    frame_index = min(frame_index, len(frames) - 1)
                    result.append(frames[frame_index].copy())
                return result
            else:
                # Upsample by repeating
                return MediaProcessor.create_frame_sequence(frames, target_count, 'loop')
        
        elif method == 'extend':
            # Extend with last frame
            result = frames.copy()
            while len(result) < target_count:
                result.append(frames[-1].copy())
            return result[:target_count]
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    @staticmethod
    def enhance_image_quality(image: np.ndarray) -> np.ndarray:
        """
        Enhance image quality using various techniques
        
        Args:
            image: Input image
            
        Returns:
            Enhanced image
        """
        try:
            # Convert to float for processing
            img_float = image.astype(np.float32) / 255.0
            
            # Slight sharpening
            kernel = np.array([[-0.1, -0.1, -0.1],
                              [-0.1,  1.8, -0.1],
                              [-0.1, -0.1, -0.1]])
            sharpened = cv2.filter2D(img_float, -1, kernel)
            
            # Ensure values are in valid range
            sharpened = np.clip(sharpened, 0, 1)
            
            # Convert back to uint8
            result = (sharpened * 255).astype(np.uint8)
            
            return result
        except:
            # Return original if enhancement fails
            return image
    
    @staticmethod
    def blend_images(background: np.ndarray, foreground: np.ndarray, 
                    mask: np.ndarray, feather: int = 2) -> np.ndarray:
        """
        Blend two images using a mask with optional feathering
        
        Args:
            background: Background image
            foreground: Foreground image
            mask: Blend mask (0-255)
            feather: Feathering amount in pixels
            
        Returns:
            Blended image
        """
        try:
            # Ensure all images have same dimensions
            h, w = background.shape[:2]
            foreground = cv2.resize(foreground, (w, h))
            mask = cv2.resize(mask, (w, h))
            
            # Apply feathering to mask
            if feather > 0:
                mask = cv2.GaussianBlur(mask, (feather * 2 + 1, feather * 2 + 1), 0)
            
            # Normalize mask
            mask_norm = mask.astype(np.float32) / 255.0
            
            # Ensure mask has 3 channels
            if len(mask_norm.shape) == 2:
                mask_norm = cv2.merge([mask_norm, mask_norm, mask_norm])
            
            # Blend images
            background_float = background.astype(np.float32)
            foreground_float = foreground.astype(np.float32)
            
            blended = background_float * (1 - mask_norm) + foreground_float * mask_norm
            
            return blended.astype(np.uint8)
        except:
            # Return background if blending fails
            return background