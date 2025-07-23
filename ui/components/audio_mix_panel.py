import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
from pathlib import Path
from core.audio_processor import AudioMixConfig, AudioProcessor
from utils.file_utils import get_supported_files

class AudioMixPanel:
    def __init__(self, parent, audio_change_callback):
        self.parent = parent
        self.audio_change_callback = audio_change_callback
        
        # Audio mix variables
        self.audio_enabled = tk.BooleanVar(value=False)
        self.audio_overlay_folder = tk.StringVar()
        self.video_audio_source = tk.StringVar(value="primary")  # primary or secondary
        self.overlay_volume_var = tk.DoubleVar(value=0.5)
        self.video_volume_var = tk.DoubleVar(value=0.7)
        self.fade_in_var = tk.DoubleVar(value=0.0)
        self.fade_out_var = tk.DoubleVar(value=0.0)
        self.normalize_var = tk.BooleanVar(value=True)
        self.overlay_start_var = tk.DoubleVar(value=0.0)
        
        # Audio processor for validation
        self.audio_processor = AudioProcessor()
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup audio mix configuration UI"""
        self.frame = ttk.LabelFrame(self.parent, text="Audio Mix Settings", padding="10")
        self.frame.columnconfigure(0, weight=1)
        
        # Enable/disable audio mix
        ttk.Checkbutton(self.frame, text="Enable Audio Mix", variable=self.audio_enabled,
                       command=self.toggle_audio_mix).grid(row=0, column=0, sticky=tk.W, pady=(0, 15))
        
        # Audio settings frame
        self.audio_settings_frame = ttk.Frame(self.frame)
        self.audio_settings_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        self.audio_settings_frame.columnconfigure(0, weight=1)
        
        # === AUDIO OVERLAY FOLDER SELECTION ===
        overlay_folder_frame = ttk.LabelFrame(self.audio_settings_frame, text="Audio Overlay Folder", padding="8")
        overlay_folder_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        overlay_folder_frame.columnconfigure(0, weight=1)
        
        # Folder input
        folder_input_frame = ttk.Frame(overlay_folder_frame)
        folder_input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 8))
        folder_input_frame.columnconfigure(0, weight=1)
        
        ttk.Entry(folder_input_frame, textvariable=self.audio_overlay_folder).grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 8))
        ttk.Button(folder_input_frame, text="Browse Folder", command=self.browse_audio_overlay_folder).grid(row=0, column=1)
        ttk.Button(folder_input_frame, text="Check Match", command=self.check_audio_matching).grid(row=0, column=2, padx=(8, 0))
        
        # Info label
        self.overlay_info_label = ttk.Label(overlay_folder_frame, text="Select folder containing audio files to overlay", 
                                           font=('Arial', 9), foreground='gray')
        self.overlay_info_label.grid(row=1, column=0, sticky=tk.W)
        
        # === VIDEO AUDIO SOURCE SELECTION ===
        video_audio_frame = ttk.LabelFrame(self.audio_settings_frame, text="Video Audio Source", padding="8")
        video_audio_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        video_audio_frame.columnconfigure(1, weight=1)
        
        ttk.Label(video_audio_frame, text="Extract audio from:").grid(row=0, column=0, sticky=tk.W, pady=(0, 8))
        
        # Radio buttons for video source
        source_radio_frame = ttk.Frame(video_audio_frame)
        source_radio_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0))
        
        ttk.Radiobutton(source_radio_frame, text="Primary Folder Videos", 
                       variable=self.video_audio_source, value="primary",
                       command=self.on_audio_change).grid(row=0, column=0, sticky=tk.W, padx=(0, 20))
        
        ttk.Radiobutton(source_radio_frame, text="Secondary Folder Videos", 
                       variable=self.video_audio_source, value="secondary",
                       command=self.on_audio_change).grid(row=0, column=1, sticky=tk.W)
        
        # Description
        ttk.Label(video_audio_frame, text="Audio will be extracted from videos in the selected folder", 
                 font=('Arial', 9), foreground='gray').grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))
        
        # === VOLUME CONTROLS ===
        volume_frame = ttk.LabelFrame(self.audio_settings_frame, text="Volume Controls", padding="8")
        volume_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        volume_frame.columnconfigure(1, weight=1)
        
        # Overlay volume
        ttk.Label(volume_frame, text="Overlay Audio Volume:").grid(row=0, column=0, sticky=tk.W, pady=(0, 8))
        overlay_vol_frame = ttk.Frame(volume_frame)
        overlay_vol_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=(0, 8))
        overlay_vol_frame.columnconfigure(0, weight=1)
        
        self.overlay_volume_scale = ttk.Scale(overlay_vol_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL, 
                                             variable=self.overlay_volume_var, command=self.on_volume_change)
        self.overlay_volume_scale.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 8))
        
        self.overlay_volume_label = ttk.Label(overlay_vol_frame, text="50%", width=6, font=('Arial', 9, 'bold'))
        self.overlay_volume_label.grid(row=0, column=1)
        
        # Video volume
        ttk.Label(volume_frame, text="Video Audio Volume:").grid(row=1, column=0, sticky=tk.W, pady=(0, 8))
        video_vol_frame = ttk.Frame(volume_frame)
        video_vol_frame.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=(0, 8))
        video_vol_frame.columnconfigure(0, weight=1)
        
        self.video_volume_scale = ttk.Scale(video_vol_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL, 
                                           variable=self.video_volume_var, command=self.on_volume_change)
        self.video_volume_scale.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 8))
        
        self.video_volume_label = ttk.Label(video_vol_frame, text="70%", width=6, font=('Arial', 9, 'bold'))
        self.video_volume_label.grid(row=0, column=1)
        
        # === OVERLAY TIMING ===
        timing_frame = ttk.LabelFrame(self.audio_settings_frame, text="Overlay Timing", padding="8")
        timing_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        timing_frame.columnconfigure(1, weight=1)
        
        ttk.Label(timing_frame, text="Overlay Start Time:").grid(row=0, column=0, sticky=tk.W, pady=(0, 8))
        overlay_time_frame = ttk.Frame(timing_frame)
        overlay_time_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=(0, 8))
        overlay_time_frame.columnconfigure(0, weight=1)
        
        self.overlay_start_scale = ttk.Scale(overlay_time_frame, from_=0.0, to=30.0, orient=tk.HORIZONTAL, 
                                            variable=self.overlay_start_var, command=self.on_overlay_time_change)
        self.overlay_start_scale.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 8))
        
        self.overlay_start_label = ttk.Label(overlay_time_frame, text="0.0s", width=6, font=('Arial', 9, 'bold'))
        self.overlay_start_label.grid(row=0, column=1)
        
        # Important note about overlay behavior
        ttk.Label(timing_frame, text="💡 Overlay Mode: Short audio overlays on long audio (no looping)", 
                 font=('Arial', 9), foreground='blue').grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))
        
        # === AUDIO EFFECTS ===
        effects_frame = ttk.LabelFrame(self.audio_settings_frame, text="Audio Effects", padding="8")
        effects_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        effects_frame.columnconfigure(1, weight=1)
        effects_frame.columnconfigure(3, weight=1)
        
        # Fade in
        ttk.Label(effects_frame, text="Fade In:").grid(row=0, column=0, sticky=tk.W, pady=(0, 8))
        fade_in_spin = ttk.Spinbox(effects_frame, from_=0.0, to=10.0, increment=0.1, 
                                  textvariable=self.fade_in_var, width=8)
        fade_in_spin.grid(row=0, column=1, sticky=tk.W, pady=(0, 8), padx=(10, 5))
        fade_in_spin.bind('<KeyRelease>', self.on_audio_change)
        fade_in_spin.bind('<<Increment>>', self.on_audio_change)
        fade_in_spin.bind('<<Decrement>>', self.on_audio_change)
        
        ttk.Label(effects_frame, text="seconds").grid(row=0, column=2, sticky=tk.W, pady=(0, 8), padx=(5, 20))
        
        # Fade out
        ttk.Label(effects_frame, text="Fade Out:").grid(row=0, column=3, sticky=tk.W, pady=(0, 8))
        fade_out_spin = ttk.Spinbox(effects_frame, from_=0.0, to=10.0, increment=0.1, 
                                   textvariable=self.fade_out_var, width=8)
        fade_out_spin.grid(row=0, column=4, sticky=tk.W, pady=(0, 8), padx=(10, 5))
        fade_out_spin.bind('<KeyRelease>', self.on_audio_change)
        fade_out_spin.bind('<<Increment>>', self.on_audio_change)
        fade_out_spin.bind('<<Decrement>>', self.on_audio_change)
        
        ttk.Label(effects_frame, text="seconds").grid(row=0, column=5, sticky=tk.W, pady=(0, 8))
        
        # Normalize audio
        ttk.Checkbutton(effects_frame, text="Normalize Audio Levels", 
                       variable=self.normalize_var, command=self.on_audio_change).grid(row=1, column=0, columnspan=6, sticky=tk.W, pady=(5, 0))
        
        # Initially disable all settings
        self.toggle_audio_mix()
    
    def toggle_audio_mix(self):
        """Toggle audio mix settings"""
        state = 'normal' if self.audio_enabled.get() else 'disabled'
        
        # Enable/disable all child widgets
        self.set_frame_state(self.audio_settings_frame, state)
        
        self.audio_change_callback()
    
    def set_frame_state(self, frame, state):
        """Recursively set state for all widgets in frame"""
        for child in frame.winfo_children():
            if isinstance(child, (ttk.Frame, ttk.LabelFrame)):
                self.set_frame_state(child, state)
            elif hasattr(child, 'configure'):
                try:
                    child.configure(state=state)
                except tk.TclError:
                    pass  # Some widgets don't support state
    
    def browse_audio_overlay_folder(self):
        """Browse for audio overlay folder"""
        folder_path = filedialog.askdirectory(title="Select Audio Overlay Folder")
        if folder_path:
            self.audio_overlay_folder.set(folder_path)
            
            # Update info label with file count
            audio_files = self.audio_processor.get_audio_files(folder_path)
            self.overlay_info_label.configure(
                text=f"✅ Found {len(audio_files)} audio files - will be matched with primary files by name"
            )
            
            self.audio_change_callback()
    
    def check_audio_matching(self):
        """Check audio file matching"""
        if not self.audio_overlay_folder.get():
            messagebox.showwarning("Warning", "Please select an audio overlay folder first")
            return
        
        try:
            audio_files = self.audio_processor.get_audio_files(self.audio_overlay_folder.get())
            
            if audio_files:
                # Show available audio files
                file_list = "\n".join([f"🎵 {Path(f).name}" for f in audio_files[:15]])
                if len(audio_files) > 15:
                    file_list += f"\n... and {len(audio_files) - 15} more files"
                
                messagebox.showinfo("Audio Overlay Files Found", 
                                   f"Found {len(audio_files)} audio files for overlay:\n\n{file_list}\n\n"
                                   "These audio files will be matched with primary files by name.\n"
                                   "Example: 'video1.mp4' matches with 'video1.mp3'")
            else:
                messagebox.showwarning("No Audio Files", 
                                     "No supported audio files found in the selected folder.\n\n"
                                     "Supported formats: MP3, WAV, AAC, M4A, OGG, FLAC, WMA")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to check audio files: {str(e)}")
    
    def on_volume_change(self, value):
        """Handle volume slider changes"""
        # Update volume labels
        overlay_vol = int(self.overlay_volume_var.get() * 100)
        video_vol = int(self.video_volume_var.get() * 100)
        
        self.overlay_volume_label.configure(text=f"{overlay_vol}%")
        self.video_volume_label.configure(text=f"{video_vol}%")
        
        self.audio_change_callback()
    
    def on_overlay_time_change(self, value):
        """Handle overlay start time change"""
        time_val = self.overlay_start_var.get()
        self.overlay_start_label.configure(text=f"{time_val:.1f}s")
        self.audio_change_callback()
    
    def on_audio_change(self, event=None):
        """Handle audio setting changes"""
        self.audio_change_callback()
    
    def get_audio_mix_config(self):
        """Get audio mix configuration"""
        config = AudioMixConfig()
        config.enabled = self.audio_enabled.get()
        config.audio_folder = self.audio_overlay_folder.get()
        config.audio_source = self.video_audio_source.get()  # primary or secondary
        config.mix_volume = self.overlay_volume_var.get()
        config.template_volume = self.video_volume_var.get()
        config.fade_in = self.fade_in_var.get()
        config.fade_out = self.fade_out_var.get()
        config.normalize = self.normalize_var.get()
        config.overlay_mode = True  # Always use overlay mode (no looping)
        config.overlay_start_time = self.overlay_start_var.get()
        config.replace_template_audio = False  # Always mix, don't replace
        return config
    
    def grid(self, **kwargs):
        """Grid the frame"""
        self.frame.grid(**kwargs)