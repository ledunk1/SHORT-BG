import cv2
import numpy as np
from moviepy.editor import VideoFileClip, ImageSequenceClip, ImageClip
import os
import tempfile
from typing import List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import time
import gc
import threading
from concurrent.futures import ThreadPoolExecutor
import subprocess
import shutil

from core.text_overlay import TextOverlayConfig, TextOverlayRenderer
from core.parallel_processor import ParallelFrameProcessor, GPUAcceleratedProcessor
from utils.gpu_detector import GPUDetector

class OptimizedGreenScreenAutoEditor:
    """Optimized Green Screen Auto Editor with parallel processing"""
    
    def __init__(self, template_path: str, video1_path: str, video2_path: Optional[str], 
                 output_path: str, text_overlay_config: Optional[TextOverlayConfig] = None,
                 filename_for_text: str = "", max_workers: Optional[int] = None):
        """
        Initialize the Optimized Green Screen Auto Editor
        
        Args:
            template_path: Path to template video/gif/image with green screen areas
            video1_path: Path to first video to be inserted
            video2_path: Path to second video to be inserted (optional)
            output_path: Path for output video
            text_overlay_config: Configuration for text overlay (optional)
            filename_for_text: Filename to use for text overlay (without extension)
            max_workers: Maximum number of worker threads (None = auto-detect)
        """
        self.template_path = template_path
        self.video1_path = video1_path
        self.video2_path = video2_path
        self.output_path = output_path
        self.text_overlay_config = text_overlay_config
        self.filename_for_text = filename_for_text
        
        # Green screen detection parameters
        self.green_lower = np.array([40, 40, 40])    # Lower HSV threshold for green
        self.green_upper = np.array([80, 255, 255])  # Upper HSV threshold for green
        self.min_area = 1000  # Minimum area for green screen detection
        
        # Initialize parallel processor
        self.parallel_processor = ParallelFrameProcessor(max_workers=max_workers, use_gpu=True)
        self.gpu_processor = GPUAcceleratedProcessor()
        
        # Initialize text overlay renderer
        if self.text_overlay_config:
            self.text_renderer = TextOverlayRenderer(self.text_overlay_config)
        else:
            self.text_renderer = None
        
        # Initialize GPU detector
        self.gpu_detector = GPUDetector()
        
        # Memory management
        self.chunk_size = 4  # Smaller chunks for RAM efficiency
        self.max_memory_frames = 30  # Reduced memory frames
    
    def detect_green_screen_areas(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect green screen areas in a frame (GPU accelerated)
        
        Args:
            frame: Input frame in BGR format
            
        Returns:
            List of bounding boxes (x, y, w, h) for detected green screen areas
        """
        # Convert BGR to HSV (GPU accelerated if available)
        if self.gpu_processor.gpu_available:
            try:
                gpu_frame = cv2.UMat(frame)
                hsv = cv2.cvtColor(gpu_frame, cv2.COLOR_BGR2HSV)
                hsv = hsv.get()  # Download from GPU
            except:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        else:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for green color
        mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        
        # Apply morphological operations (GPU accelerated)
        mask = self.gpu_processor.morphology_gpu(mask, cv2.MORPH_CLOSE, 5)
        mask = self.gpu_processor.morphology_gpu(mask, cv2.MORPH_OPEN, 5)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area and get bounding boxes
        green_areas = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_area:
                x, y, w, h = cv2.boundingRect(contour)
                green_areas.append((x, y, w, h))
        
        # Sort by area (largest first) to ensure consistent ordering
        green_areas.sort(key=lambda box: box[2] * box[3], reverse=True)
        
        return green_areas
    
    def get_media_info(self, media_path: str) -> dict:
        """Get media information (optimized)"""
        info = {
            'is_image': False,
            'is_video': False,
            'duration': 0,
            'fps': 30,
            'width': 0,
            'height': 0,
            'frame_count': 1
        }
        
        if media_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # Static image
            info['is_image'] = True
            try:
                image = cv2.imread(media_path)
                if image is not None:
                    info['height'], info['width'] = image.shape[:2]
            except:
                pass
        else:
            # Video/GIF
            info['is_video'] = True
            try:
                cap = cv2.VideoCapture(media_path)
                info['fps'] = cap.get(cv2.CAP_PROP_FPS) or 30
                info['frame_count'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                info['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                info['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                info['duration'] = info['frame_count'] / info['fps']
                cap.release()
            except:
                pass
        
        return info
    
    def load_frames_optimized(self, media_path: str, target_frame_count: int, 
                             media_type: str = "primary") -> List[np.ndarray]:
        """
        FIXED: Load frames - ALL must sync to PRIMARY duration
        
        Args:
            media_path: Path to media file
            target_frame_count: Target number of frames (based on PRIMARY duration)
            media_type: Type of media for logging
            
        Returns:
            List of frames synced to PRIMARY duration
        """
        print(f"  📥 Loading {media_type} media: {Path(media_path).name} (sync to PRIMARY duration)")
        
        info = self.get_media_info(media_path)
        frames = []
        
        if info['is_image']:
            # Static image - repeat for PRIMARY duration
            print(f"    📷 Static image - repeating for PRIMARY duration ({target_frame_count} frames)")
            image = cv2.imread(media_path)
            if image is not None:
                frames = [image for _ in range(target_frame_count)]
        else:
            # Video - MUST sync to PRIMARY duration
            print(f"    🎥 Video: {info['frame_count']} frames → {target_frame_count} frames (PRIMARY duration)")
            
            cap = cv2.VideoCapture(media_path)
            original_frames = []
            
            # Load all frames first
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                original_frames.append(frame)
                frame_count += 1
            
            cap.release()
            
            if original_frames:
                # Synchronize to PRIMARY duration
                if len(original_frames) == target_frame_count:
                    print(f"      ✅ Perfect match with PRIMARY duration")
                    frames = original_frames
                elif len(original_frames) < target_frame_count:
                    # Video is shorter than PRIMARY - LOOP to match PRIMARY
                    loops_needed = (target_frame_count + len(original_frames) - 1) // len(original_frames)
                    print(f"      🔄 LOOPING {loops_needed} times to match PRIMARY duration")
                    frames = []
                    for i in range(target_frame_count):
                        frame_index = i % len(original_frames)
                        frames.append(original_frames[frame_index].copy())
                else:
                    # Video is longer than PRIMARY - CUT to match PRIMARY
                    print(f"      ✂️ CUTTING to match PRIMARY duration")
                    frames = []
                    step = len(original_frames) / target_frame_count
                    for i in range(target_frame_count):
                        frame_index = int(i * step)
                        frame_index = min(frame_index, len(original_frames) - 1)
                        frames.append(original_frames[frame_index].copy())
        
        print(f"    ✅ {media_type} synced to PRIMARY: {len(frames)} frames")
        return frames
    
    def process_single_frame(self, frame_data: Tuple, frame_idx: int) -> Optional[np.ndarray]:
        """
        Process a single frame (optimized for parallel execution)
        
        Args:
            frame_data: Tuple containing (template_frame, video1_frame, video2_frame, green_areas)
            frame_idx: Frame index
            
        Returns:
            Processed frame or None if error
        """
        try:
            template_frame, video1_frame, video2_frame, green_areas = frame_data
            
            # Work on a copy to avoid modifying original
            current_frame = template_frame.copy()
            
            # Replace green screen areas
            for area_index, green_area in enumerate(green_areas):
                if area_index == 0 and video1_frame is not None:
                    # Use first video for first green area
                    replacement_frame = self.resize_frame_to_fit_green_area_gpu(video1_frame, green_area)
                    current_frame = self.apply_replacement_gpu(current_frame, replacement_frame, green_area)
                elif area_index == 1 and video2_frame is not None:
                    # Use second video for second green area
                    replacement_frame = self.resize_frame_to_fit_green_area_gpu(video2_frame, green_area)
                    current_frame = self.apply_replacement_gpu(current_frame, replacement_frame, green_area)
            
            # Apply text overlay
            if self.text_renderer and self.filename_for_text:
                current_frame = self.text_renderer.render_text(current_frame, self.filename_for_text, frame_idx)
            
            # Convert to RGB for moviepy
            current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
            
            return current_frame
            
        except Exception as e:
            print(f"    ❌ Frame {frame_idx} processing error: {e}")
            return None
    
    def resize_frame_to_fit_green_area_gpu(self, frame: np.ndarray, 
                                          green_area: Tuple[int, int, int, int]) -> np.ndarray:
        """GPU-accelerated frame resizing"""
        x, y, w, h = green_area
        return self.gpu_processor.resize_gpu(frame, (w, h))
    
    def create_mask_from_green_area_gpu(self, frame: np.ndarray, 
                                       green_box: Tuple[int, int, int, int]) -> np.ndarray:
        """GPU-accelerated mask creation"""
        x, y, w, h = green_box
        
        # Create region of interest
        roi = frame[y:y+h, x:x+w]
        
        # Convert to HSV (GPU accelerated)
        if self.gpu_processor.gpu_available:
            try:
                gpu_roi = cv2.UMat(roi)
                hsv_roi = cv2.cvtColor(gpu_roi, cv2.COLOR_BGR2HSV)
                hsv_roi = hsv_roi.get()
            except:
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        else:
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Create mask for green color in ROI
        mask_roi = cv2.inRange(hsv_roi, self.green_lower, self.green_upper)
        
        # Apply morphological operations (GPU accelerated)
        mask_roi = self.gpu_processor.morphology_gpu(mask_roi, cv2.MORPH_CLOSE, 3)
        mask_roi = self.gpu_processor.morphology_gpu(mask_roi, cv2.MORPH_OPEN, 3)
        
        # Smooth the mask (GPU accelerated)
        mask_roi = self.gpu_processor.gaussian_blur_gpu(mask_roi, 5)
        
        # Create full-size mask
        full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        full_mask[y:y+h, x:x+w] = mask_roi
        
        return full_mask
    
    def apply_replacement_gpu(self, template_frame: np.ndarray, replacement_frame: np.ndarray, 
                             green_area: Tuple[int, int, int, int]) -> np.ndarray:
        """GPU-accelerated replacement application"""
        x, y, w, h = green_area
        
        # Create mask for this green area
        mask = self.create_mask_from_green_area_gpu(template_frame, green_area)
        
        # Create positioned replacement frame
        positioned_replacement = np.zeros_like(template_frame)
        positioned_replacement[y:y+h, x:x+w] = replacement_frame
        
        # Apply replacement using GPU-accelerated blending
        result = self.gpu_processor.blend_gpu(template_frame, positioned_replacement, mask)
        
        return result
    
    def process_video_optimized(self, progress_callback=None):
        """
        Optimized main processing function with parallel processing
        
        Args:
            progress_callback: Optional callback function for progress updates
        """
        print("\n" + "🚀" * 20 + " OPTIMIZED VIDEO PROCESSING START " + "🚀" * 20)
        start_time = time.time()
        
        # Print system information
        self.gpu_detector.print_gpu_info()
        self.gpu_detector.enable_opencv_acceleration()
        
        print(f"\n📁 Processing Files:")
        print(f"  🎭 Template: {Path(self.template_path).name}")
        print(f"  🎥 Primary: {Path(self.video1_path).name}")
        if self.video2_path:
            print(f"  🎬 Secondary: {Path(self.video2_path).name}")
        print(f"  💾 Output: {Path(self.output_path).name}")
        
        # Load template and analyze
        print(f"\n🔍 Analyzing template...")
        template_frames, template_info = self.load_template_frames()
        
        if not template_frames:
            raise ValueError("Template is empty or could not be loaded")
        
        print(f"  ✅ Template loaded: {len(template_frames)} frames")
        
        # Detect green screen areas from first frame
        print(f"\n🟢 Detecting green screen areas...")
        green_areas = self.detect_green_screen_areas(template_frames[0])
        
        print(f"  ✅ Detected {len(green_areas)} green screen areas:")
        for i, (x, y, w, h) in enumerate(green_areas):
            print(f"    🎯 Area {i+1}: ({x}, {y}) size {w}x{h}")
        
        # FIXED: Duration analysis - PRIMARY VIDEO CONTROLS EVERYTHING
        print(f"\n⏱️ Duration Analysis:")
        primary_info = self.get_media_info(self.video1_path)
        
        # PRIMARY VIDEO ALWAYS CONTROLS DURATION - NO EXCEPTIONS
        if primary_info['is_video'] and primary_info['duration'] > 0:
            # PRIMARY video controls everything
            target_duration = primary_info['duration']
            target_fps = primary_info['fps']
            print(f"  ✅ PRIMARY VIDEO controls duration: {target_duration:.2f}s at {target_fps} FPS")
            print(f"  📏 Template will be cut/looped to match PRIMARY duration")
            print(f"  📏 Secondary will be cut/looped to match PRIMARY duration")
        else:
            # Primary is image - use default duration, but still primary controls
            target_duration = 5.0  # Default for images
            target_fps = 30
            print(f"  ✅ PRIMARY is image - using default: {target_duration:.2f}s at {target_fps} FPS")
            print(f"  📏 Template will be cut/looped to match PRIMARY duration")
            print(f"  📏 Secondary will be cut/looped to match PRIMARY duration")
        
        target_frame_count = int(target_duration * target_fps)
        print(f"  🎯 Target frame count: {target_frame_count} frames")
        
        # Load media files - ALL MUST FOLLOW PRIMARY DURATION
        print(f"\n📥 Loading Media Files (ALL synced to PRIMARY duration: {target_duration:.2f}s):")
        load_start = time.time()
        
        # Load primary video first
        video1_frames = self.load_frames_optimized(self.video1_path, target_frame_count, "primary")
        
        # Load secondary video - MUST sync to PRIMARY duration
        video2_frames = []
        if self.video2_path and len(green_areas) > 1:
            video2_frames = self.load_frames_optimized(self.video2_path, target_frame_count, "secondary")
        
        # Prepare template frames - MUST sync to PRIMARY duration
        template_frames = self.prepare_template_frames_optimized(template_frames, template_info, target_frame_count)
        
        load_time = time.time() - load_start
        print(f"  ⏱️ All media loaded in {load_time:.2f}s")
        
        # Prepare frame data for parallel processing
        print(f"\n🎨 Preparing Frame Data:")
        frames_data = []
        
        for i in range(target_frame_count):
            template_frame = template_frames[i] if i < len(template_frames) else template_frames[-1]
            video1_frame = video1_frames[i] if i < len(video1_frames) else None
            video2_frame = video2_frames[i] if i < len(video2_frames) else None
            
            frames_data.append((template_frame, video1_frame, video2_frame, green_areas))
        
        print(f"  ✅ Prepared {len(frames_data)} frame data entries")
        
        # Process frames in parallel
        print(f"\n🚀 Parallel Frame Processing:")
        processing_start = time.time()
        
        # Use pipeline processing for better memory management
        processed_frames = self.parallel_processor.process_frames_pipeline(
            frames_data,
            self.process_single_frame,
            progress_callback,
            buffer_size=self.chunk_size * 2
        )
        
        # Filter out None results and process in streaming mode
        print(f"  🔍 Filtering valid frames...")
        valid_frames = []
        for i, frame in enumerate(processed_frames):
            if frame is not None:
                valid_frames.append(frame)
            
            # Memory cleanup every 50 frames
            if i % 50 == 0:
                gc.collect()
        
        processing_time = time.time() - processing_start
        print(f"  ✅ Parallel processing complete in {processing_time:.2f}s")
        print(f"  📊 Valid frames: {len(valid_frames)}/{len(processed_frames)}")
        
        # Clear processed frames from memory
        del processed_frames
        gc.collect()
        
        # Create final video
        if valid_frames:
            print(f"\n🎞️ Creating Final Video:")
            encoding_start = time.time()
            
            # Use streaming approach to save RAM
            self.create_video_streaming(valid_frames, target_fps, target_duration)
            
            encoding_time = time.time() - encoding_start
            print(f"  ✅ Video encoded in {encoding_time:.2f}s")
            
            # Force garbage collection
            gc.collect()
            
            if progress_callback:
                progress_callback(1.0)
            
            total_time = time.time() - start_time
            print(f"\n🎉 SUCCESS! Optimized video processing complete!")
            print(f"  📁 Output saved to: {self.output_path}")
            print(f"  ⏱️ Total processing time: {total_time:.2f}s")
            print(f"  📊 Average processing speed: {len(valid_frames) / total_time:.1f} fps")
            print(f"  🚀 Estimated speedup: {self.parallel_processor.max_workers}x")
            print("🚀" * 60)
        
        else:
            raise ValueError("No valid output frames generated")
    
    def create_video_streaming(self, frames: List[np.ndarray], fps: float, duration: float):
        """
        Create video using streaming approach to save RAM
        
        Args:
            frames: List of processed frames
            fps: Target FPS
            duration: Target duration
        """
        print(f"  🎬 Using streaming encoding (RAM efficient)")
        
        # Get frame dimensions
        if not frames:
            raise ValueError("No frames to encode")
        
        height, width = frames[0].shape[:2]
        
        # Setup video writer with OpenCV (more reliable than moviepy)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        temp_video_path = self.output_path.replace('.mp4', '_temp.mp4')
        
        print(f"  📹 Creating video: {width}x{height} @ {fps} FPS")
        
        # Create video writer
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            raise Exception("Failed to open video writer")
        
        # Write frames in chunks to save RAM
        chunk_size = 30  # Process 30 frames at a time
        total_frames = len(frames)
        
        print(f"  💾 Writing {total_frames} frames in chunks of {chunk_size}...")
        
        for i in range(0, total_frames, chunk_size):
            chunk_end = min(i + chunk_size, total_frames)
            chunk_frames = frames[i:chunk_end]
            
            # Write chunk frames
            for j, frame in enumerate(chunk_frames):
                # Convert RGB back to BGR for OpenCV
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(bgr_frame)
                
                # Progress update
                if (i + j + 1) % 60 == 0:  # Every 2 seconds at 30fps
                    progress = (i + j + 1) / total_frames * 100
                    print(f"    📊 Writing progress: {progress:.1f}%")
            
            # Clear chunk from memory
            del chunk_frames
            gc.collect()
        
        # Release video writer
        out.release()
        print(f"  ✅ Video frames written to temporary file")
        
        # Add audio using FFmpeg directly (more reliable)
        self.add_audio_with_ffmpeg(temp_video_path, self.output_path, duration)
        
        # Clean up temporary file
        try:
            os.remove(temp_video_path)
            print(f"  🗑️ Temporary file cleaned up")
        except:
            pass
    
    def add_audio_with_ffmpeg(self, video_path: str, output_path: str, target_duration: float):
        """
        Add audio using FFmpeg directly for better reliability
        
        Args:
            video_path: Path to video file (without audio)
            output_path: Final output path
            target_duration: Target duration
        """
        print(f"  🔊 Adding audio from PRIMARY video...")
        
        try:
            # Check if primary video has audio
            primary_info = self.get_media_info(self.video1_path)
            
            # First check if primary video actually has audio
            check_cmd = [
                'ffprobe', '-v', 'quiet', '-select_streams', 'a:0', 
                '-show_entries', 'stream=codec_name', '-of', 'csv=p=0', 
                self.video1_path
            ]
            
            try:
                audio_check = subprocess.run(check_cmd, capture_output=True, text=True, timeout=30)
                has_audio = bool(audio_check.stdout.strip())
            except:
                has_audio = False
            
            if not has_audio:
                print(f"    ⚠️ PRIMARY video has no audio, copying video only...")
                shutil.copy2(video_path, output_path)
                return
            
            # Build FFmpeg command
            cmd = [
                'ffmpeg', '-y',  # Overwrite output
                '-i', video_path,  # Input video (no audio)
                '-i', self.video1_path,  # Input audio source
                '-c:v', 'copy',  # Copy video stream
                '-c:a', 'aac',   # Encode audio as AAC
                '-map', '0:v:0',  # Map video from first input
                '-map', '1:a:0',  # Map audio from second input
                '-shortest',      # Use shortest stream duration
                '-avoid_negative_ts', 'make_zero',
                '-loglevel', 'error',  # Reduce FFmpeg output
                output_path
            ]
            
            print(f"    🎵 Extracting audio from: {Path(self.video1_path).name}")
            
            # Run FFmpeg
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"    ✅ Audio successfully added from PRIMARY video")
            else:
                print(f"    ⚠️ FFmpeg audio failed, creating video without audio...")
                shutil.copy2(video_path, output_path)
                
        except subprocess.TimeoutExpired:
            print(f"    ⚠️ FFmpeg timeout, creating video without audio...")
            shutil.copy2(video_path, output_path)
            
        except Exception as e:
            print(f"    ⚠️ Audio processing failed: {e}")
            print(f"    📹 Creating video without audio...")
            shutil.copy2(video_path, output_path)
    
    def load_template_frames(self) -> Tuple[List[np.ndarray], dict]:
        """Load template frames from file"""
        frames = []
        info = self.get_media_info(self.template_path)
        
        if info['is_image']:
            image = cv2.imread(self.template_path)
            if image is not None:
                frames.append(image)
        else:
            cap = cv2.VideoCapture(self.template_path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()
        
        return frames, info
    
    def prepare_template_frames_optimized(self, template_frames: List[np.ndarray], 
                                        template_info: dict, target_frame_count: int) -> List[np.ndarray]:
        """Prepare template frames optimized"""
        if template_info['is_image']:
            # FIXED: Template image - repeat for PRIMARY duration
            print(f"    📷 Template is image - repeating for PRIMARY duration ({target_frame_count} frames)")
            return [template_frames[0] for _ in range(target_frame_count)]
        
        original_frame_count = len(template_frames)
        print(f"    🎭 Template video: {original_frame_count} → {target_frame_count} frames (PRIMARY duration)")
        
        if original_frame_count == target_frame_count:
            print(f"      ✅ Template perfect match with PRIMARY duration")
            return template_frames
        elif original_frame_count < target_frame_count:
            # Template is shorter than PRIMARY - LOOP to match PRIMARY
            loops_needed = (target_frame_count + original_frame_count - 1) // original_frame_count
            print(f"      🔄 Template LOOPING {loops_needed} times to match PRIMARY duration")
            adjusted_frames = []
            for i in range(target_frame_count):
                frame_index = i % original_frame_count
                adjusted_frames.append(template_frames[frame_index].copy())
            return adjusted_frames
        else:
            # Template is longer than PRIMARY - CUT to match PRIMARY
            print(f"      ✂️ Template CUTTING to match PRIMARY duration")
            return template_frames[:target_frame_count]