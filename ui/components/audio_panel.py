import tkinter as tk
from tkinter import ttk, messagebox
from core.audio_processor import AudioProcessor

class AudioPanel:
    def __init__(self, parent, primary_folder, secondary_folder, audio_change_callback):
        self.parent = parent
        self.primary_folder = primary_folder
        self.secondary_folder = secondary_folder
        self.audio_change_callback = audio_change_callback
        
        # Audio processor
        self.audio_processor = AudioProcessor()
        
        # Audio configuration variables
        self.audio_enabled = tk.BooleanVar(value=False)
        self.audio_mode = tk.StringVar(value="none")  # 'none', 'original', 'mix'
        self.original_source = tk.StringVar(value="primary")  # 'primary' or 'secondary'
        self.mix_main_source = tk.StringVar(value="primary")  # 'primary' or 'secondary'
        self.mix_folder_source = tk.StringVar(value="secondary")  # Which folder to use for overlay
        self.mix_volume_main = tk.DoubleVar(value=0.7)  # Volume for main audio (base)
        self.mix_volume_overlay = tk.DoubleVar(value=0.3)  # Volume for overlay audio
        self.loop_audio = tk.BooleanVar(value=True)  # Whether to loop audio if shorter than video
        
        # Flag to prevent callback during initialization
        self._initializing = True
        
        self.setup_ui()
        
        # Enable callbacks after initialization
        self._initializing = False
    
    def setup_ui(self):
        """Setup audio configuration UI"""
        self.frame = ttk.LabelFrame(self.parent, text="Audio Settings", padding="10")
        self.frame.columnconfigure(0, weight=1)
        
        # Enable/disable audio
        ttk.Checkbutton(self.frame, text="Enable Audio Processing", variable=self.audio_enabled,
                       command=self.toggle_audio).grid(row=0, column=0, sticky=tk.W, pady=(0, 10))
        
        # Audio settings frame
        self.audio_settings_frame = ttk.Frame(self.frame)
        self.audio_settings_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        self.audio_settings_frame.columnconfigure(0, weight=1)
        
        # Audio mode selection
        mode_frame = ttk.LabelFrame(self.audio_settings_frame, text="Audio Mode", padding="5")
        mode_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # None option
        ttk.Radiobutton(mode_frame, text="No Audio", variable=self.audio_mode, value="none",
                       command=self.on_mode_change).grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        # Original audio option
        ttk.Radiobutton(mode_frame, text="Use Original Audio", variable=self.audio_mode, value="original",
                       command=self.on_mode_change).grid(row=1, column=0, sticky=tk.W, pady=(0, 5))
        
        # Mix audio option
        ttk.Radiobutton(mode_frame, text="Mix Audio from Both Folders", variable=self.audio_mode, value="mix",
                       command=self.on_mode_change).grid(row=2, column=0, sticky=tk.W)
        
        # Original audio settings
        self.original_frame = ttk.LabelFrame(self.audio_settings_frame, text="Original Audio Settings", padding="5")
        self.original_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(self.original_frame, text="Audio Source:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        source_frame = ttk.Frame(self.original_frame)
        source_frame.grid(row=0, column=1, sticky=tk.W, padx=(10, 0), pady=(0, 5))
        
        ttk.Radiobutton(source_frame, text="Primary Folder", variable=self.original_source, value="primary",
                       command=self.on_audio_change).grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        ttk.Radiobutton(source_frame, text="Secondary Folder", variable=self.original_source, value="secondary",
                       command=self.on_audio_change).grid(row=0, column=1, sticky=tk.W)
        
        # Mix audio settings
        self.mix_frame = ttk.LabelFrame(self.audio_settings_frame, text="Mix Audio Settings", padding="5")
        self.mix_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        self.mix_frame.columnconfigure(1, weight=1)
        
        # Main audio source (base audio from filename matching)
        ttk.Label(self.mix_frame, text="Base Audio Source:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        main_source_frame = ttk.Frame(self.mix_frame)
        main_source_frame.grid(row=0, column=1, sticky=tk.W, padx=(10, 0), pady=(0, 5))
        
        ttk.Radiobutton(main_source_frame, text="Primary Folder", variable=self.mix_main_source, value="primary",
                       command=self.on_audio_change).grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        ttk.Radiobutton(main_source_frame, text="Secondary Folder", variable=self.mix_main_source, value="secondary",
                       command=self.on_audio_change).grid(row=0, column=1, sticky=tk.W)
        
        # Mix folder source (overlay audio)
        ttk.Label(self.mix_frame, text="Overlay Audio Source:").grid(row=1, column=0, sticky=tk.W, pady=(10, 5))
        
        mix_folder_frame = ttk.Frame(self.mix_frame)
        mix_folder_frame.grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=(10, 5))
        
        ttk.Radiobutton(mix_folder_frame, text="Primary Folder (filename matching)", variable=self.mix_folder_source, value="primary",
                       command=self.on_audio_change).grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        ttk.Radiobutton(mix_folder_frame, text="Secondary Folder (filename matching)", variable=self.mix_folder_source, value="secondary",
                       command=self.on_audio_change).grid(row=0, column=1, sticky=tk.W)
        
        # Add explanation for overlay audio
        overlay_info = ttk.Label(self.mix_frame, 
                                text="💡 Overlay audio will be matched by filename from selected folder", 
                                font=('Arial', 8), foreground='blue')
        overlay_info.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(2, 0))
        
        # Volume controls
        ttk.Label(self.mix_frame, text="Base Audio Volume:").grid(row=3, column=0, sticky=tk.W, pady=(10, 5))
        
        main_volume_frame = ttk.Frame(self.mix_frame)
        main_volume_frame.grid(row=3, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=(10, 5))
        main_volume_frame.columnconfigure(0, weight=1)
        
        self.main_volume_scale = ttk.Scale(main_volume_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL,
                                          variable=self.mix_volume_main, command=self.on_volume_change)
        self.main_volume_scale.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        
        self.main_volume_label = ttk.Label(main_volume_frame, text="70%")
        self.main_volume_label.grid(row=0, column=1)
        
        ttk.Label(self.mix_frame, text="Overlay Audio Volume:").grid(row=4, column=0, sticky=tk.W, pady=(0, 5))
        
        overlay_volume_frame = ttk.Frame(self.mix_frame)
        overlay_volume_frame.grid(row=4, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=(0, 5))
        overlay_volume_frame.columnconfigure(0, weight=1)
        
        self.overlay_volume_scale = ttk.Scale(overlay_volume_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL,
                                             variable=self.mix_volume_overlay, command=self.on_volume_change)
        self.overlay_volume_scale.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        
        self.overlay_volume_label = ttk.Label(overlay_volume_frame, text="30%")
        self.overlay_volume_label.grid(row=0, column=1)
        
        # Audio info frame
        self.info_frame = ttk.LabelFrame(self.audio_settings_frame, text="Audio Information", padding="5")
        self.info_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.info_text = tk.Text(self.info_frame, height=4, width=50, wrap=tk.WORD, state='disabled')
        self.info_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Control buttons
        button_frame = ttk.Frame(self.audio_settings_frame)
        button_frame.grid(row=4, column=0, pady=(0, 10))
        
        ttk.Button(button_frame, text="Check Audio Files", command=self.check_audio_files).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(button_frame, text="Test Audio Match", command=self.test_audio_match).grid(row=0, column=1)
        
        # Audio behavior settings
        behavior_frame = ttk.LabelFrame(self.audio_settings_frame, text="Audio Behavior", padding="5")
        behavior_frame.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Checkbutton(behavior_frame, text="Loop audio if shorter than video", 
                       variable=self.loop_audio, command=self.on_audio_change).grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        # Info about looping
        loop_info = ttk.Label(behavior_frame, 
                             text="⚠️ If disabled: Mixed audio stops when it ends, continues with base audio only", 
                             font=('Arial', 8), foreground='orange')
        loop_info.grid(row=1, column=0, sticky=tk.W, pady=(0, 5))
        
        # Example
        example_info = ttk.Label(behavior_frame, 
                                text="Example: 3s mixed + 10s video = Mixed 0-3s, base audio only 3-10s", 
                                font=('Arial', 8), foreground='gray')
        example_info.grid(row=2, column=0, sticky=tk.W)
        
        # Initialize UI state
        self.toggle_audio()
    
    def toggle_audio(self):
        """Toggle audio settings"""
        state = 'normal' if self.audio_enabled.get() else 'disabled'
        self.set_frame_state(self.audio_settings_frame, state)
        self.on_mode_change()
        self.on_audio_change()
    
    def set_frame_state(self, frame, state):
        """Recursively set state for all widgets in frame"""
        for child in frame.winfo_children():
            if isinstance(child, (ttk.Frame, ttk.LabelFrame)):
                self.set_frame_state(child, state)
            elif hasattr(child, 'configure'):
                try:
                    child.configure(state=state)
                except tk.TclError:
                    pass
    
    def on_mode_change(self):
        """Handle audio mode change"""
        if not self.audio_enabled.get():
            return
        
        mode = self.audio_mode.get()
        
        # Show/hide relevant frames
        if mode == "original":
            self.original_frame.grid()
            self.mix_frame.grid_remove()
        elif mode == "mix":
            self.original_frame.grid_remove()
            self.mix_frame.grid()
        else:  # none
            self.original_frame.grid_remove()
            self.mix_frame.grid_remove()
        
        self.update_audio_info()
        self.on_audio_change()
    
    def on_audio_change(self):
        """Handle audio setting changes"""
        self.update_audio_info()
        
        # Only call callback if not initializing
        if not self._initializing:
            self.audio_change_callback()
    
    def on_volume_change(self, value=None):
        """Handle volume slider changes"""
        main_volume = self.mix_volume_main.get()
        overlay_volume = self.mix_volume_overlay.get()
        
        self.main_volume_label.configure(text=f"{int(main_volume * 100)}%")
        self.overlay_volume_label.configure(text=f"{int(overlay_volume * 100)}%")
        
        self.on_audio_change()
    
    def update_audio_info(self):
        """Update audio information display"""
        if not self.audio_enabled.get():
            info_text = "Audio processing is disabled."
        else:
            config = self.get_audio_config()
            audio_info = self.audio_processor.get_audio_info(
                config, 
                self.primary_folder.get(), 
                self.secondary_folder.get()
            )
            
            mode = config.get('mode', 'none')
            
            if mode == 'none':
                info_text = "No audio will be added to output videos."
            
            elif mode == 'original':
                source = config.get('original_source', 'primary')
                folder = 'Primary' if source == 'primary' else 'Secondary'
                file_count = audio_info.get('primary_files_count' if source == 'primary' else 'secondary_files_count', 0)
                info_text = f"Using original audio from {folder} folder.\n"
                info_text += f"Found {file_count} media files with audio."
            
            elif mode == 'mix':
                base_source = config.get('mix_main_source', 'primary')
                overlay_source = config.get('mix_folder_source', 'secondary')
                base_folder = 'Primary' if base_source == 'primary' else 'Secondary'
                overlay_folder = 'Primary' if overlay_source == 'primary' else 'Secondary'
                
                main_volume = int(config.get('mix_volume_main', 0.7) * 100)
                overlay_volume = int(config.get('mix_volume_overlay', 0.3) * 100)
                
                info_text = f"Mixing audio: {base_folder} base ({main_volume}%) + {overlay_folder} overlay ({overlay_volume}%)\n"
                info_text += f"Primary files: {audio_info.get('primary_files_count', 0)}\n"
                info_text += f"Secondary files: {audio_info.get('secondary_files_count', 0)}\n"
                info_text += f"Matched pairs: {audio_info.get('matched_pairs_count', 0)}\n"
                info_text += f"📝 Base: {base_folder} folder (filename matching)\n"
                info_text += f"🎵 Overlay: {overlay_folder} folder (filename matching)\n"
                info_text += f"🔄 Both audios matched by video filename"
            
            # Add looping info
            if mode in ['original', 'mix']:
                loop_enabled = config.get('loop_audio', True)
                loop_status = "✅ Enabled" if loop_enabled else "❌ Disabled"
                info_text += f"\nAudio Looping: {loop_status}"
                
                if not loop_enabled:
                    if mode == 'mix':
                        info_text += "\n⚠️ Mixed audio will stop, continues with base audio only"
                    else:
                        info_text += "\n⚠️ Audio will play naturally without looping"
            
            if not audio_info.get('valid', False):
                info_text += "\n⚠️ Configuration is invalid!"
        
        # Update text widget
        self.info_text.configure(state='normal')
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, info_text)
        self.info_text.configure(state='disabled')
    
    def check_audio_files(self):
        """Check audio files in folders"""
        primary_folder = self.primary_folder.get()
        secondary_folder = self.secondary_folder.get()
        
        if not primary_folder and not secondary_folder:
            messagebox.showwarning("Warning", "Please select primary and/or secondary folders first.")
            return
        
        info_lines = []
        
        if primary_folder:
            primary_files = self.audio_processor.get_audio_files_from_folder(primary_folder)
            info_lines.append(f"Primary folder: {len(primary_files)} media files")
            
            if primary_files:
                info_lines.append("Primary files:")
                for i, file_path in enumerate(primary_files[:10], 1):  # Show first 10
                    info_lines.append(f"  {i}. {Path(file_path).name}")
                if len(primary_files) > 10:
                    info_lines.append(f"  ... and {len(primary_files) - 10} more files")
        
        if secondary_folder:
            secondary_files = self.audio_processor.get_audio_files_from_folder(secondary_folder)
            info_lines.append(f"\nSecondary folder: {len(secondary_files)} media files")
            
            if secondary_files:
                info_lines.append("Secondary files:")
                for i, file_path in enumerate(secondary_files[:10], 1):  # Show first 10
                    info_lines.append(f"  {i}. {Path(file_path).name}")
                if len(secondary_files) > 10:
                    info_lines.append(f"  ... and {len(secondary_files) - 10} more files")
        
        messagebox.showinfo("Audio Files Information", "\n".join(info_lines))
    
    def test_audio_match(self):
        """Test audio file matching"""
        primary_folder = self.primary_folder.get()
        secondary_folder = self.secondary_folder.get()
        
        if not primary_folder or not secondary_folder:
            messagebox.showwarning("Warning", "Please select both primary and secondary folders.")
            return
        
        matched_pairs = self.audio_processor.match_audio_files(primary_folder, secondary_folder)
        
        if matched_pairs:
            info_lines = [f"Found {len(matched_pairs)} matching audio file pairs:\n"]
            
            for i, (primary, secondary) in enumerate(matched_pairs[:10], 1):  # Show first 10
                info_lines.append(f"{i}. {Path(primary).name} ↔ {Path(secondary).name}")
            
            if len(matched_pairs) > 10:
                info_lines.append(f"\n... and {len(matched_pairs) - 10} more pairs")
            
            messagebox.showinfo("Audio Matching Results", "\n".join(info_lines))
        else:
            messagebox.showwarning("No Matches", 
                                 "No matching audio files found.\n\n"
                                 "Make sure files have the same base name:\n"
                                 "• Primary: 'audio1.mp4'\n"
                                 "• Secondary: 'audio1.wav' ✅")
    
    def get_audio_config(self):
        """Get audio configuration"""
        return {
            'enabled': self.audio_enabled.get(),
            'mode': self.audio_mode.get(),
            'original_source': self.original_source.get(),
            'mix_main_source': self.mix_main_source.get(),
            'mix_folder_source': self.mix_folder_source.get(),
            'mix_volume_main': self.mix_volume_main.get(),
            'mix_volume_overlay': self.mix_volume_overlay.get(),
            'loop_audio': self.loop_audio.get(),
            'primary_folder': self.primary_folder.get(),
            'secondary_folder': self.secondary_folder.get()
        }
    
    def grid(self, **kwargs):
        """Grid the frame"""
        self.frame.grid(**kwargs)