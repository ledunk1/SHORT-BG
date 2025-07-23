import os
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from moviepy.editor import AudioFileClip, CompositeAudioClip
import librosa
import soundfile as sf

class AudioProcessor:
    """Audio processing and mixing utilities"""
    
    # Supported audio formats
    SUPPORTED_AUDIO_EXTENSIONS = {'.mp3', '.wav', '.aac', '.m4a', '.ogg', '.flac', '.wma'}
    
    def __init__(self):
        self.audio_cache = {}  # Cache for loaded audio clips
    
    @staticmethod
    def is_supported_audio_file(file_path: str) -> bool:
        """Check if file is a supported audio format"""
        return Path(file_path).suffix.lower() in AudioProcessor.SUPPORTED_AUDIO_EXTENSIONS
    
    @staticmethod
    def get_audio_files(folder_path: str) -> List[str]:
        """
        Get list of supported audio files in folder
        
        Args:
            folder_path: Path to folder containing audio files
            
        Returns:
            List of audio file paths
        """
        if not os.path.exists(folder_path):
            return []
        
        audio_files = []
        
        try:
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                
                # Skip directories
                if os.path.isdir(file_path):
                    continue
                
                # Check if file is supported audio
                if AudioProcessor.is_supported_audio_file(file_path):
                    audio_files.append(file_path)
            
            # Sort files alphabetically
            audio_files.sort()
            
        except PermissionError:
            print(f"Permission denied accessing audio folder: {folder_path}")
        except Exception as e:
            print(f"Error reading audio folder {folder_path}: {e}")
        
        return audio_files
    
    @staticmethod
    def match_audio_files(primary_files: List[str], audio_files: List[str]) -> List[Tuple[str, Optional[str]]]:
        """
        Match primary files with audio files by base name
        
        Args:
            primary_files: List of primary video/image file paths
            audio_files: List of audio file paths
            
        Returns:
            List of tuples (primary_file, matched_audio_file or None)
        """
        matched_pairs = []
        
        # Create mapping of base names to audio files
        audio_map = {}
        for audio_file in audio_files:
            base_name = Path(audio_file).stem.lower()  # Case insensitive
            audio_map[base_name] = audio_file
        
        # Match primary files with audio files
        for primary_file in primary_files:
            primary_base = Path(primary_file).stem.lower()  # Case insensitive
            
            if primary_base in audio_map:
                audio_file = audio_map[primary_base]
                matched_pairs.append((primary_file, audio_file))
                print(f"🎵 Audio matched: {Path(primary_file).name} ↔ {Path(audio_file).name}")
            else:
                # No matching audio file found
                matched_pairs.append((primary_file, None))
                print(f"🔇 No audio match for: {Path(primary_file).name}")
        
        return matched_pairs
    
    def get_audio_info(self, audio_path: str) -> Dict:
        """
        Get audio file information
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with audio information
        """
        info = {
            'duration': 0.0,
            'sample_rate': 44100,
            'channels': 2,
            'format': 'unknown',
            'bitrate': 0,
            'exists': os.path.exists(audio_path)
        }
        
        if not info['exists']:
            return info
        
        try:
            # Use librosa for detailed audio info
            y, sr = librosa.load(audio_path, sr=None)
            info['duration'] = len(y) / sr
            info['sample_rate'] = sr
            info['channels'] = 1 if len(y.shape) == 1 else y.shape[0]
            info['format'] = Path(audio_path).suffix.lower()
            
            # Estimate bitrate (rough calculation)
            file_size = os.path.getsize(audio_path)
            info['bitrate'] = int((file_size * 8) / info['duration'] / 1000) if info['duration'] > 0 else 0
            
        except Exception as e:
            print(f"Error getting audio info for {audio_path}: {e}")
            # Fallback to moviepy
            try:
                with AudioFileClip(audio_path) as clip:
                    info['duration'] = clip.duration
                    info['sample_rate'] = clip.fps
                    info['channels'] = clip.nchannels if hasattr(clip, 'nchannels') else 2
            except:
                pass
        
        return info
    
    def load_audio_clip(self, audio_path: str, target_duration: float = None) -> Optional[AudioFileClip]:
        """
        Load audio clip with optional duration adjustment (supports overlay mode)
        
        Args:
            audio_path: Path to audio file
            target_duration: Target duration in seconds (None = keep original, used for overlay base)
            
        Returns:
            AudioFileClip or None if failed
        """
        try:
            # Check cache first
            cache_key = f"{audio_path}_{target_duration}"
            if cache_key in self.audio_cache:
                return self.audio_cache[cache_key]
            
            # Load audio clip
            audio_clip = AudioFileClip(audio_path)
            
            # Keep original duration for overlay mode
            # Duration adjustment will be handled in mixing functions
            
            # Cache the clip
            self.audio_cache[cache_key] = audio_clip
            
            return audio_clip
            
        except Exception as e:
            print(f"Error loading audio clip {audio_path}: {e}")
            return None
    
    def extract_audio_from_video(self, video_path: str) -> Optional[AudioFileClip]:
        """
        Extract audio from video file
        
        Args:
            video_path: Path to video file
            
        Returns:
            AudioFileClip or None if no audio or failed
        """
        try:
            from moviepy.editor import VideoFileClip
            
            with VideoFileClip(video_path) as video_clip:
                if video_clip.audio is not None:
                    # Extract audio and cache it
                    audio_clip = video_clip.audio
                    cache_key = f"video_audio_{video_path}"
                    self.audio_cache[cache_key] = audio_clip
                    return audio_clip
                else:
                    print(f"  🔇 No audio track in video: {Path(video_path).name}")
                    return None
                    
        except Exception as e:
            print(f"  ❌ Error extracting audio from video {video_path}: {e}")
            return None
    
    def get_audio_from_source(self, source_type: str, file_path: str, 
                             audio_folder: str = None, current_filename: str = None) -> Optional[AudioFileClip]:
        """
        Get audio from specified source
        
        Args:
            source_type: "primary" or "secondary" (for video audio extraction)
            file_path: Path to primary/secondary file
            audio_folder: Path to audio folder (for overlay audio)
            current_filename: Current processing filename for matching
            
        Returns:
            AudioFileClip or None
        """
        try:
            if source_type == "primary":
                # Extract audio from primary video
                print(f"  🎵 Extracting audio from primary: {Path(file_path).name}")
                return self.extract_audio_from_video(file_path)
                
            elif source_type == "secondary":
                # Extract audio from secondary video
                print(f"  🎵 Extracting audio from secondary: {Path(file_path).name}")
                return self.extract_audio_from_video(file_path)
                
            else:
                print(f"  ❌ Unknown audio source type: {source_type}")
                return None
                
        except Exception as e:
            print(f"  ❌ Error getting audio from source {source_type}: {e}")
            return None
    
    def get_overlay_audio_for_file(self, audio_folder: str, current_filename: str) -> Optional[AudioFileClip]:
        """
        Get overlay audio file that matches current processing file
        
        Args:
            audio_folder: Path to audio overlay folder
            current_filename: Current file being processed (without extension)
            
        Returns:
            AudioFileClip or None
        """
        try:
            if not audio_folder or not os.path.exists(audio_folder):
                return None
            
            audio_files = self.get_audio_files(audio_folder)
            if not audio_files:
                return None
            
            # Find matching audio file by name
            file_name = current_filename.lower()
            for audio_file in audio_files:
                audio_name = Path(audio_file).stem.lower()
                if audio_name == file_name:
                    print(f"  🎵 Found overlay audio: {Path(audio_file).name}")
                    return self.load_audio_clip(audio_file)
            
            print(f"  🔇 No overlay audio found for: {current_filename}")
            return None
            
        except Exception as e:
            print(f"  ❌ Error getting overlay audio: {e}")
            return None
    
    def overlay_audio_on_template(self, template_audio: AudioFileClip, 
                                 overlay_audio: AudioFileClip, 
                                 overlay_volume: float = 0.5,
                                 template_volume: float = 0.7,
                                 overlay_start_time: float = 0.0,
                                 overlay_mode: bool = True) -> AudioFileClip:
        """
        Overlay audio on template audio (no looping, just overlay)
        
        Args:
            template_audio: Original template audio
            overlay_audio: Audio to overlay
            overlay_volume: Volume level for overlay audio (0.0 - 1.0)
            template_volume: Volume level for template audio (0.0 - 1.0)
            overlay_start_time: When to start overlay (seconds)
            overlay_mode: True = overlay (no extend), False = loop/extend
            
        Returns:
            Audio with overlay applied
        """
        try:
            print(f"  🎵 Audio overlay mode: {'Overlay' if overlay_mode else 'Loop/Extend'}")
            print(f"  ⏱️ Template duration: {template_audio.duration:.2f}s")
            print(f"  🎶 Overlay duration: {overlay_audio.duration:.2f}s")
            print(f"  🕐 Overlay start time: {overlay_start_time:.2f}s")
            
            # Adjust template volume
            template_adjusted = template_audio.volumex(template_volume)
            
            if overlay_mode:
                # OVERLAY MODE: Audio pendek overlay di atas audio panjang
                # Tidak ada looping, audio pendek hanya ditumpuk
                
                # Adjust overlay volume
                overlay_adjusted = overlay_audio.volumex(overlay_volume)
                
                # Set start time for overlay
                if overlay_start_time > 0:
                    overlay_adjusted = overlay_adjusted.set_start(overlay_start_time)
                
                # Create composite - overlay will only play for its duration
                result_audio = CompositeAudioClip([template_adjusted, overlay_adjusted])
                
                # Ensure result has same duration as template (longer audio)
                result_audio = result_audio.set_duration(template_audio.duration)
                
                print(f"  ✅ Overlay applied: {overlay_audio.duration:.2f}s overlay on {template_audio.duration:.2f}s base")
                
            else:
                # LOOP/EXTEND MODE: Traditional mixing with looping
                target_duration = template_audio.duration
                
                if overlay_audio.duration < target_duration:
                    # Audio is shorter - loop it
                    loops_needed = int(np.ceil(target_duration / overlay_audio.duration))
                    overlay_extended = overlay_audio.loop(loops_needed).subclip(0, target_duration)
                elif overlay_audio.duration > target_duration:
                    # Audio is longer - trim it
                    overlay_extended = overlay_audio.subclip(0, target_duration)
                else:
                    # Same duration
                    overlay_extended = overlay_audio
                
                # Adjust volumes
                overlay_adjusted = overlay_extended.volumex(overlay_volume)
                
                # Create composite
                result_audio = CompositeAudioClip([template_adjusted, overlay_adjusted])
                
                print(f"  ✅ Loop/extend applied: {overlay_audio.duration:.2f}s → {target_duration:.2f}s")
            
            return result_audio
            
        except Exception as e:
            print(f"  ❌ Error overlaying audio: {e}")
            return template_audio  # Return original if mixing fails
    
    def apply_audio_effects(self, audio_clip: AudioFileClip, 
                           fade_in: float = 0.0, 
                           fade_out: float = 0.0,
                           normalize: bool = True) -> AudioFileClip:
        """
        Apply audio effects
        
        Args:
            audio_clip: Input audio clip
            fade_in: Fade in duration in seconds
            fade_out: Fade out duration in seconds
            normalize: Whether to normalize audio levels
            
        Returns:
            Audio clip with effects applied
        """
        try:
            result = audio_clip
            
            # Apply fade in
            if fade_in > 0:
                result = result.audio_fadein(fade_in)
            
            # Apply fade out
            if fade_out > 0:
                result = result.audio_fadeout(fade_out)
            
            # Normalize audio (basic volume adjustment)
            if normalize:
                # Simple normalization by reducing volume slightly to prevent clipping
                result = result.volumex(0.9)
            
            return result
            
        except Exception as e:
            print(f"Error applying audio effects: {e}")
            return audio_clip
    
    def cleanup_cache(self):
        """Clean up cached audio clips"""
        for clip in self.audio_cache.values():
            try:
                if hasattr(clip, 'close'):
                    clip.close()
            except:
                pass
        self.audio_cache.clear()
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.cleanup_cache()


class AudioMixConfig:
    """Configuration for audio mixing"""
    
    def __init__(self):
        self.enabled = False
        self.audio_folder = ""
        self.audio_source = "folder"  # "folder", "primary", or "secondary"
        self.mix_volume = 0.5  # Volume for mixed audio (0.0 - 1.0)
        self.template_volume = 0.7  # Volume for template audio (0.0 - 1.0)
        self.fade_in = 0.0  # Fade in duration in seconds
        self.fade_out = 0.0  # Fade out duration in seconds
        self.normalize = True  # Whether to normalize audio
        self.replace_template_audio = False  # Replace template audio instead of mixing
        self.overlay_mode = True  # True = overlay (no loop), False = extend/loop
        self.overlay_start_time = 0.0  # When to start overlay (seconds)