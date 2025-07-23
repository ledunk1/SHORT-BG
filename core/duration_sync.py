import cv2
import numpy as np
from typing import List, Tuple, Dict
from pathlib import Path
from utils.media_utils import MediaProcessor

class DurationSynchronizer:
    """Handles duration synchronization between different media files"""
    
    def __init__(self):
        self.media_processor = MediaProcessor()
    
    def analyze_media_durations(self, template_path: str, primary_path: str, 
                               secondary_path: str = None) -> Dict:
        """
        Analyze durations of all media files
        
        Args:
            template_path: Path to template file
            primary_path: Path to primary media file
            secondary_path: Path to secondary media file (optional)
            
        Returns:
            Dictionary with duration analysis
        """
        analysis = {
            'template': self.media_processor.get_media_info(template_path),
            'primary': self.media_processor.get_media_info(primary_path),
            'secondary': None,
            'target_duration': 0.0,
            'target_fps': 30.0,
            'sync_strategy': 'template_based'
        }
        
        if secondary_path:
            analysis['secondary'] = self.media_processor.get_media_info(secondary_path)
        
        # Determine target duration and FPS
        template_info = analysis['template']
        primary_info = analysis['primary']
        
        if template_info['type'] == 'image':
            # Template is static image - use primary media duration
            if primary_info['type'] == 'video' and primary_info['duration'] > 0:
                analysis['target_duration'] = primary_info['duration']
                analysis['target_fps'] = primary_info['fps']
                analysis['sync_strategy'] = 'primary_based'
            else:
                # Both are images or primary has no duration - use default
                analysis['target_duration'] = 5.0  # 5 seconds default
                analysis['target_fps'] = 30.0
                analysis['sync_strategy'] = 'default'
        else:
            # Template is video - use template duration
            analysis['target_duration'] = template_info['duration']
            analysis['target_fps'] = template_info['fps']
            analysis['sync_strategy'] = 'template_based'
        
        return analysis
    
    def synchronize_frames(self, frames: List[np.ndarray], current_duration: float, 
                          current_fps: float, target_duration: float, 
                          target_fps: float, media_type: str = 'video') -> List[np.ndarray]:
        """
        Synchronize frames to target duration and FPS
        
        Args:
            frames: Input frames
            current_duration: Current duration in seconds
            current_fps: Current FPS
            target_duration: Target duration in seconds
            target_fps: Target FPS
            media_type: Type of media ('image' or 'video')
            
        Returns:
            Synchronized frames
        """
        if not frames:
            return []
        
        target_frame_count = int(target_duration * target_fps)
        
        if media_type == 'image':
            # Static image - repeat for target duration
            return [frames[0].copy() for _ in range(target_frame_count)]
        
        # Video synchronization
        current_frame_count = len(frames)
        
        if current_frame_count == target_frame_count:
            # Perfect match
            return frames.copy()
        
        elif current_frame_count < target_frame_count:
            # Video is shorter - loop/repeat
            return self._loop_frames(frames, target_frame_count)
        
        else:
            # Video is longer - sample frames
            return self._sample_frames(frames, target_frame_count)
    
    def _loop_frames(self, frames: List[np.ndarray], target_count: int) -> List[np.ndarray]:
        """
        Loop frames to reach target count
        
        Args:
            frames: Input frames
            target_count: Target frame count
            
        Returns:
            Looped frames
        """
        result = []
        for i in range(target_count):
            frame_index = i % len(frames)
            result.append(frames[frame_index].copy())
        return result
    
    def _sample_frames(self, frames: List[np.ndarray], target_count: int) -> List[np.ndarray]:
        """
        Sample frames evenly to reach target count
        
        Args:
            frames: Input frames
            target_count: Target frame count
            
        Returns:
            Sampled frames
        """
        result = []
        step = len(frames) / target_count
        
        for i in range(target_count):
            frame_index = int(i * step)
            frame_index = min(frame_index, len(frames) - 1)
            result.append(frames[frame_index].copy())
        
        return result
    
    def create_smooth_transitions(self, frames: List[np.ndarray], 
                                 transition_frames: int = 5) -> List[np.ndarray]:
        """
        Create smooth transitions when looping video
        
        Args:
            frames: Input frames
            transition_frames: Number of frames for transition
            
        Returns:
            Frames with smooth transitions
        """
        if len(frames) <= transition_frames * 2:
            return frames
        
        result = frames.copy()
        
        # Create transition at the end to beginning
        for i in range(transition_frames):
            alpha = i / transition_frames
            
            # Blend last frames with first frames
            end_frame = frames[-(transition_frames - i)]
            start_frame = frames[i]
            
            blended = cv2.addWeighted(end_frame, 1 - alpha, start_frame, alpha, 0)
            result[-(transition_frames - i)] = blended
        
        return result
    
    def optimize_frame_timing(self, frames: List[np.ndarray], target_fps: float, 
                             source_fps: float = None) -> List[np.ndarray]:
        """
        Optimize frame timing for smooth playback
        
        Args:
            frames: Input frames
            target_fps: Target FPS
            source_fps: Source FPS (if known)
            
        Returns:
            Optimized frames
        """
        if not source_fps or source_fps == target_fps:
            return frames
        
        # Calculate frame ratio
        ratio = target_fps / source_fps
        
        if ratio > 1:
            # Need more frames - interpolate
            return self._interpolate_frames(frames, ratio)
        else:
            # Need fewer frames - decimate
            return self._decimate_frames(frames, ratio)
    
    def _interpolate_frames(self, frames: List[np.ndarray], ratio: float) -> List[np.ndarray]:
        """
        Interpolate frames to increase frame rate
        
        Args:
            frames: Input frames
            ratio: Interpolation ratio
            
        Returns:
            Interpolated frames
        """
        result = []
        
        for i in range(len(frames) - 1):
            result.append(frames[i])
            
            # Add interpolated frames
            num_interpolated = int(ratio) - 1
            for j in range(num_interpolated):
                alpha = (j + 1) / (num_interpolated + 1)
                interpolated = cv2.addWeighted(frames[i], 1 - alpha, frames[i + 1], alpha, 0)
                result.append(interpolated)
        
        # Add last frame
        result.append(frames[-1])
        
        return result
    
    def _decimate_frames(self, frames: List[np.ndarray], ratio: float) -> List[np.ndarray]:
        """
        Decimate frames to decrease frame rate
        
        Args:
            frames: Input frames
            ratio: Decimation ratio
            
        Returns:
            Decimated frames
        """
        step = 1 / ratio
        result = []
        
        for i in range(len(frames)):
            if i % int(step) == 0:
                result.append(frames[i])
        
        return result