import cv2
import numpy as np
from moviepy.editor import VideoFileClip, ImageSequenceClip, ImageClip
import os
import tempfile
from typing import List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import time

from core.text_overlay import TextOverlayConfig, TextOverlayRenderer
from utils.gpu_detector import GPUDetector
from core.optimized_video_processor import OptimizedGreenScreenAutoEditor

class GreenScreenAutoEditor:
    def __init__(self, template_path: str, video1_path: str, video2_path: Optional[str], 
                 output_path: str, text_overlay_config: Optional[TextOverlayConfig] = None,
                 filename_for_text: str = ""):
        """
        Initialize the Green Screen Auto Editor
        
        Args:
            template_path: Path to template video/gif/image with green screen areas
            video1_path: Path to first video to be inserted
            video2_path: Path to second video to be inserted (optional)
            output_path: Path for output video
            text_overlay_config: Configuration for text overlay (optional)
            filename_for_text: Filename to use for text overlay (without extension)
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
        
        # Initialize text overlay renderer
        if self.text_overlay_config:
            self.text_renderer = TextOverlayRenderer(self.text_overlay_config)
        else:
            self.text_renderer = None
        
        # Initialize GPU detector
        self.gpu_detector = GPUDetector()
        
        # Add optimized processor option
        self.use_optimized = True  # Enable optimized processing by default
        
    def detect_green_screen_areas(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect green screen areas in a frame
        
        Args:
            frame: Input frame in BGR format
            
        Returns:
            List of bounding boxes (x, y, w, h) for detected green screen areas
        """
        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for green color
        mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
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
        """
        Get media information (duration, fps, dimensions, type)
        
        Args:
            media_path: Path to media file
            
        Returns:
            Dictionary with media information
        """
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
    
    def load_template_frames(self) -> Tuple[List[np.ndarray], dict]:
        """
        Load template frames from file
        
        Returns:
            Tuple of (frames list, media info)
        """
        frames = []
        info = self.get_media_info(self.template_path)
        
        if info['is_image']:
            # Static image
            image = cv2.imread(self.template_path)
            if image is not None:
                frames.append(image)
        else:
            # Video/GIF
            cap = cv2.VideoCapture(self.template_path)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            
            cap.release()
        
        return frames, info
    
    def load_media_frames(self, media_path: str, target_duration: float, target_fps: float) -> List[np.ndarray]:
        """
        Load media frames with duration synchronization
        
        Args:
            media_path: Path to media file
            target_duration: Target duration in seconds
            target_fps: Target FPS
            
        Returns:
            List of frames synchronized to target duration
        """
        frames = []
        info = self.get_media_info(media_path)
        target_frame_count = int(target_duration * target_fps)
        
        if info['is_image']:
            # Static image - repeat for target duration
            image = cv2.imread(media_path)
            if image is not None:
                frames = [image.copy() for _ in range(target_frame_count)]
        else:
            # Video/GIF - load all frames first
            cap = cv2.VideoCapture(media_path)
            original_frames = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                original_frames.append(frame)
            
            cap.release()
            
            if original_frames:
                # Synchronize duration
                if len(original_frames) == target_frame_count:
                    # Perfect match
                    frames = original_frames
                elif len(original_frames) < target_frame_count:
                    # Video is shorter - repeat/loop
                    frames = []
                    for i in range(target_frame_count):
                        frame_index = i % len(original_frames)
                        frames.append(original_frames[frame_index].copy())
                else:
                    # Video is longer - sample frames evenly
                    frames = []
                    step = len(original_frames) / target_frame_count
                    for i in range(target_frame_count):
                        frame_index = int(i * step)
                        frame_index = min(frame_index, len(original_frames) - 1)
                        frames.append(original_frames[frame_index].copy())
        
        return frames
    
    def resize_frame_to_fit_green_area(self, frame: np.ndarray, green_area: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Resize frame to exactly fit green screen area
        
        Args:
            frame: Input frame
            green_area: Green screen area (x, y, w, h)
            
        Returns:
            Resized frame that exactly fits the green area
        """
        x, y, w, h = green_area
        
        # Resize frame to exactly match green area dimensions
        resized_frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LANCZOS4)
        
        return resized_frame
    
    def create_mask_from_green_area(self, frame: np.ndarray, green_box: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Create a mask for a specific green screen area
        
        Args:
            frame: Input frame
            green_box: Bounding box (x, y, w, h) of green screen area
            
        Returns:
            Binary mask for the green screen area
        """
        x, y, w, h = green_box
        
        # Create region of interest
        roi = frame[y:y+h, x:x+w]
        
        # Convert to HSV
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Create mask for green color in ROI
        mask_roi = cv2.inRange(hsv_roi, self.green_lower, self.green_upper)
        
        # Apply morphological operations
        kernel = np.ones((3, 3), np.uint8)
        mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_CLOSE, kernel)
        mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_OPEN, kernel)
        
        # Smooth the mask
        mask_roi = cv2.GaussianBlur(mask_roi, (5, 5), 0)
        
        # Create full-size mask
        full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        full_mask[y:y+h, x:x+w] = mask_roi
        
        return full_mask
    
    def apply_text_overlay(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        """
        Apply text overlay to frame
        
        Args:
            frame: Input frame
            frame_index: Frame index for animation
            
        Returns:
            Frame with text overlay
        """
        if not self.text_renderer or not self.filename_for_text:
            return frame
        
        return self.text_renderer.render_text(frame, self.filename_for_text, frame_index)
    
    def process_video(self, progress_callback=None):
        """
        Main processing function to create the final video with HD quality
        
        Args:
            progress_callback: Optional callback function for progress updates
        """
        # Use optimized processor if enabled
        if self.use_optimized:
            print("🚀 Using optimized parallel processing...")
            optimized_processor = OptimizedGreenScreenAutoEditor(
                template_path=self.template_path,
                video1_path=self.video1_path,
                video2_path=self.video2_path,
                output_path=self.output_path,
                text_overlay_config=self.text_overlay_config,
                filename_for_text=self.filename_for_text
            )
            return optimized_processor.process_video_optimized(progress_callback)
        
        # Original processing method (fallback)
        print("\n" + "🎬" * 20 + " VIDEO PROCESSING START " + "🎬" * 20)
        start_time = time.time()
        
        # Print GPU information
        self.gpu_detector.print_gpu_info()
        
        # Enable OpenCV acceleration
        self.gpu_detector.enable_opencv_acceleration()
        
        print(f"\n📁 Processing Files:")
        print(f"  🎭 Template: {Path(self.template_path).name}")
        print(f"  🎥 Primary: {Path(self.video1_path).name}")
        if self.video2_path:
            print(f"  🎬 Secondary: {Path(self.video2_path).name}")
        print(f"  💾 Output: {Path(self.output_path).name}")
        
        # Load template frames and info
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
        
        # STEP 1: PRIMARY VIDEO CONTROLS DURATION (NO SPEED CHANGES)
        print(f"\n⏱️ Duration Analysis:")
        primary_info = self.get_media_info(self.video1_path)
        
        if primary_info['is_video'] and primary_info['duration'] > 0:
            # Primary video is the master - everything follows its duration
            target_duration = primary_info['duration']
            target_fps = primary_info['fps']
            print(f"  ✅ PRIMARY controls duration: {target_duration:.2f}s at {target_fps} FPS")
        elif template_info['is_video'] and template_info['duration'] > 0:
            # Fallback: template controls if primary is image
            target_duration = template_info['duration']
            target_fps = template_info['fps']
            print(f"  ✅ TEMPLATE controls duration: {target_duration:.2f}s at {target_fps} FPS")
        else:
            # Both are images - use default
            target_duration = 5.0  # Default 5 seconds
            target_fps = 30
            print(f"  ✅ DEFAULT duration: {target_duration:.2f}s at {target_fps} FPS")
        
        # STEP 2: Load PRIMARY video (no changes - keep original timing)
        print(f"\n📥 Loading Media Files:")
        load_start = time.time()
        video1_frames = self.load_primary_media_optimized(self.video1_path, target_duration, target_fps)
        load_primary_time = time.time() - load_start
        print(f"  ⏱️ Primary loaded in {load_primary_time:.2f}s")
        
        # STEP 3: Load SECONDARY video (adapt to primary duration)
        video2_frames = []
        if self.video2_path and len(green_areas) > 1:
            load_start = time.time()
            video2_frames = self.load_secondary_media_optimized(self.video2_path, target_duration, target_fps)
            load_secondary_time = time.time() - load_start
            print(f"  ⏱️ Secondary loaded in {load_secondary_time:.2f}s")
        
        # STEP 4: Prepare TEMPLATE frames (adapt to primary duration)
        print(f"\n🎭 Preparing Template:")
        template_start = time.time()
        target_frame_count = int(target_duration * target_fps)
        template_frames = self.prepare_template_frames_optimized(template_frames, template_info, target_frame_count)
        template_time = time.time() - template_start
        print(f"  ⏱️ Template prepared in {template_time:.2f}s")
        
        # Process each frame
        print(f"\n🎨 Frame Processing:")
        print(f"  📊 Total frames to process: {len(template_frames)}")
        processing_start = time.time()
        output_frames = []
        total_frames = len(template_frames)
        
        # Progress tracking
        last_progress_time = time.time()
        frames_processed = 0
        
        for frame_index in range(total_frames):
            frame_start = time.time()
            
            if progress_callback:
                progress_callback(frame_index / total_frames)
            
            # Get current template frame
            current_frame = template_frames[frame_index].copy()
            
            # Replace green screen areas
            for area_index, green_area in enumerate(green_areas):
                if area_index == 0 and video1_frames and frame_index < len(video1_frames):
                    # Use first video for first green area
                    replacement_frame = video1_frames[frame_index]
                    replacement_frame = self.resize_frame_to_fit_green_area(replacement_frame, green_area)
                elif area_index == 1 and video2_frames and frame_index < len(video2_frames):
                    # Use second video for second green area
                    replacement_frame = video2_frames[frame_index]
                    replacement_frame = self.resize_frame_to_fit_green_area(replacement_frame, green_area)
                else:
                    continue
                
                # Apply replacement
                current_frame = self.apply_replacement(current_frame, replacement_frame, green_area)
            
            # Apply text overlay
            current_frame = self.apply_text_overlay(current_frame, frame_index)
            
            # Convert to RGB for moviepy
            current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
            output_frames.append(current_frame)
            
            frames_processed += 1
            
            # Progress logging every 30 frames or 5 seconds
            current_time = time.time()
            if (frame_index % 30 == 0 or 
                current_time - last_progress_time >= 5.0 or 
                frame_index == total_frames - 1):
                
                progress_percent = (frame_index + 1) / total_frames * 100
                elapsed = current_time - processing_start
                fps_current = frames_processed / elapsed if elapsed > 0 else 0
                eta = (total_frames - frame_index - 1) / fps_current if fps_current > 0 else 0
                
                print(f"  🎬 Progress: {progress_percent:.1f}% "
                      f"({frame_index + 1}/{total_frames}) "
                      f"| Speed: {fps_current:.1f} fps "
                      f"| ETA: {eta:.1f}s")
                last_progress_time = current_time
        
        # Create high-quality video from processed frames
        if output_frames:
            print(f"\n🎞️ Creating Final Video:")
            encoding_start = time.time()
            
            # Get optimal encoder settings
            encoder_settings = self.gpu_detector.get_optimal_encoder_settings()
            print(f"  🎯 Using encoder: {encoder_settings['codec']}")
            print(f"  ⚡ Hardware acceleration: {'✅ Enabled' if encoder_settings['hardware_acceleration'] else '❌ Disabled'}")
            
            output_clip = ImageSequenceClip(output_frames, fps=target_fps)
            
            # Add audio from template if available and not a static image
            # FIXED: Add audio from PRIMARY video, not template
            if not primary_info['is_image']:
                print(f"  🔊 Processing audio...")
                try:
                    primary_clip = VideoFileClip(self.video1_path)
                    if primary_clip.audio is not None:
                        audio = primary_clip.audio
                        
                        # FIXED: No looping - use original audio duration
                        if audio.duration > target_duration:
                            # Trim audio if longer than video
                            audio = audio.subclip(0, target_duration)
                            print(f"    ✂️ Audio trimmed to {target_duration:.2f}s")
                        elif audio.duration < target_duration:
                            # If audio is shorter, keep original without looping
                            print(f"    ⚠️ Audio ({audio.duration:.2f}s) shorter than video ({target_duration:.2f}s)")
                            print(f"    🔇 Keeping original audio without looping")
                        
                        output_clip = output_clip.set_audio(audio)
                        print(f"  ✅ Audio from PRIMARY video added")
                    else:
                        print(f"  ⚠️ PRIMARY video has no audio")
                    primary_clip.close()
                except Exception as e:
                    print(f"  ⚠️ Warning: Could not add audio from primary video: {e}")
                    # Continue without audio if there's an error
                    pass
            
            # Write output video with high quality settings
            print(f"  💾 Encoding video...")
            
            try:
                # Prepare FFmpeg parameters
                ffmpeg_params = ['-pix_fmt', 'yuv420p']
                
                # Use software encoding for stability
                if encoder_settings['hardware_acceleration']:
                    print(f"    ⚠️ Using software encoding for stability")
                    codec = 'libx264'
                    ffmpeg_params.extend(['-crf', '18'])
                else:
                    codec = encoder_settings['codec']
                    ffmpeg_params.extend(['-crf', encoder_settings['crf']])
                
                # Write video with error handling
                output_clip.write_videofile(
                    self.output_path, 
                    codec=codec,
                    audio_codec='aac' if output_clip.audio else None,
                    bitrate='8000k',
                    preset='medium',
                    ffmpeg_params=ffmpeg_params,
                    verbose=False,  # Reduce verbosity
                    logger=None     # Disable logger
                )
                
            except Exception as encoding_error:
                print(f"    ❌ Encoding failed: {encoding_error}")
                print(f"    🔄 Retrying with basic settings...")
                
                try:
                    # Fallback encoding
                    output_clip.write_videofile(
                        self.output_path,
                        codec='libx264',
                        audio_codec='aac' if output_clip.audio else None,
                        bitrate='5000k',
                        preset='fast',
                        ffmpeg_params=['-pix_fmt', 'yuv420p'],
                        verbose=False,
                        logger=None
                    )
                    print(f"    ✅ Fallback encoding successful")
                    
                except Exception as fallback_error:
                    print(f"    ❌ Fallback encoding failed: {fallback_error}")
                    raise Exception(f"Video encoding failed: {fallback_error}")
            
            encoding_time = time.time() - encoding_start
            print(f"  ✅ Video encoded in {encoding_time:.2f}s")
            
            # Clean up
            output_clip.close()
            
            if progress_callback:
                progress_callback(1.0)
            
            total_time = time.time() - start_time
            print(f"\n🎉 SUCCESS! Video processing complete!")
            print(f"  📁 Output saved to: {self.output_path}")
            print(f"  ⏱️ Total processing time: {total_time:.2f}s")
            print(f"  📊 Average processing speed: {total_frames / total_time:.1f} fps")
            print("🎬" * 60)
        
        else:
            raise ValueError("No output frames generated")
    
    def load_primary_media_optimized(self, media_path: str, target_duration: float, target_fps: float) -> List[np.ndarray]:
        """
        Load PRIMARY media - NO SPEED CHANGES, keep original timing
        If primary is shorter than target, loop it
        If primary is longer than target, cut it
        """
        frames = []
        info = self.get_media_info(media_path)
        target_frame_count = int(target_duration * target_fps)
        
        if info['is_image']:
            # Static image - repeat for target duration
            print(f"    📷 Primary is image - repeating for {target_duration:.2f}s")
            image = cv2.imread(media_path)
            if image is not None:
                frames = [image.copy() for _ in range(target_frame_count)]
        else:
            # Video - load frames efficiently
            print(f"    🎥 Primary video: {info['duration']:.2f}s → {target_duration:.2f}s")
            
            # Load all frames first
            cap = cv2.VideoCapture(media_path)
            original_frames = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                original_frames.append(frame)
            cap.release()
            
            if original_frames:
                original_frame_count = len(original_frames)
                
                if original_frame_count == target_frame_count:
                    # Perfect match - no changes needed
                    print(f"      ✅ Perfect match: {original_frame_count} frames")
                    frames = original_frames
                    
                elif original_frame_count < target_frame_count:
                    # Primary is shorter - LOOP/REPEAT (no speed change)
                    loops_needed = (target_frame_count + original_frame_count - 1) // original_frame_count
                    print(f"      🔄 Looping {loops_needed} times: {original_frame_count} → {target_frame_count} frames")
                    
                    frames = []
                    for i in range(target_frame_count):
                        frame_index = i % original_frame_count
                        frames.append(original_frames[frame_index].copy())
                        
                else:
                    # Primary is longer - CUT to target duration (no speed change)
                    print(f"      ✂️ Cutting: {original_frame_count} → {target_frame_count} frames")
                    frames = original_frames[:target_frame_count]
        
        print(f"    ✅ Primary loaded: {len(frames)} frames")
        return frames
    
    def load_secondary_media_optimized(self, media_path: str, target_duration: float, target_fps: float) -> List[np.ndarray]:
        """
        Load SECONDARY media - adapt to primary duration
        Same logic as primary: loop if shorter, cut if longer
        """
        frames = []
        info = self.get_media_info(media_path)
        target_frame_count = int(target_duration * target_fps)
        
        if info['is_image']:
            # Static image - repeat for target duration
            print(f"    📷 Secondary is image - repeating for {target_duration:.2f}s")
            image = cv2.imread(media_path)
            if image is not None:
                frames = [image.copy() for _ in range(target_frame_count)]
        else:
            # Video - adapt to primary duration
            print(f"    🎬 Secondary video: {info['duration']:.2f}s → {target_duration:.2f}s")
            
            # Load all frames
            cap = cv2.VideoCapture(media_path)
            original_frames = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                original_frames.append(frame)
            cap.release()
            
            if original_frames:
                original_frame_count = len(original_frames)
                
                if original_frame_count <= target_frame_count:
                    # Secondary is shorter or equal - LOOP/REPEAT
                    print(f"      🔄 Secondary looping: {original_frame_count} → {target_frame_count} frames")
                    frames = []
                    for i in range(target_frame_count):
                        frame_index = i % original_frame_count
                        frames.append(original_frames[frame_index].copy())
                else:
                    # Secondary is longer - CUT
                    print(f"      ✂️ Secondary cutting: {original_frame_count} → {target_frame_count} frames")
                    frames = original_frames[:target_frame_count]
        
        print(f"    ✅ Secondary loaded: {len(frames)} frames")
        return frames
    
    def prepare_template_frames_optimized(self, template_frames: List[np.ndarray], 
                                        template_info: dict, target_frame_count: int) -> List[np.ndarray]:
        """
        Prepare template frames - adapt to primary duration
        Same logic: loop if shorter, cut if longer
        """
        if template_info['is_image']:
            # Static template - repeat for target duration
            print(f"    📷 Template is image - repeating for {target_frame_count} frames")
            return [template_frames[0].copy() for _ in range(target_frame_count)]
        
        # Video template
        original_frame_count = len(template_frames)
        print(f"    🎭 Template video: {original_frame_count} → {target_frame_count} frames")
        
        if original_frame_count == target_frame_count:
            # Perfect match
            print(f"      ✅ Template perfect match")
            return template_frames
            
        elif original_frame_count < target_frame_count:
            # Template is shorter - LOOP/REPEAT
            print(f"      🔄 Template looping: {original_frame_count} → {target_frame_count} frames")
            adjusted_frames = []
            for i in range(target_frame_count):
                frame_index = i % original_frame_count
                adjusted_frames.append(template_frames[frame_index].copy())
            return adjusted_frames
            
        else:
            # Template is longer - CUT
            print(f"      ✂️ Template cutting: {original_frame_count} → {target_frame_count} frames")
            return template_frames[:target_frame_count]
    
    def apply_replacement(self, template_frame: np.ndarray, replacement_frame: np.ndarray, 
                         green_area: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Apply replacement frame to green screen area
        
        Args:
            template_frame: Template frame
            replacement_frame: Replacement frame (already resized to fit green area)
            green_area: Green screen area (x, y, w, h)
            
        Returns:
            Frame with replacement applied
        """
        x, y, w, h = green_area
        
        # Create mask for this green area
        mask = self.create_mask_from_green_area(template_frame, green_area)
        
        # Convert mask to 3-channel and normalize
        mask_3d = cv2.merge([mask, mask, mask]).astype(float) / 255.0
        
        # Create positioned replacement frame
        positioned_replacement = np.zeros_like(template_frame)
        positioned_replacement[y:y+h, x:x+w] = replacement_frame
        
        # Apply replacement using mask with smooth blending
        result = template_frame.astype(float) * (1 - mask_3d) + positioned_replacement.astype(float) * mask_3d
        
        return result.astype(np.uint8)