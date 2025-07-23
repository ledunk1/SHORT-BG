import tkinter as tk
from tkinter import ttk, messagebox
import threading
import os
from pathlib import Path
from core.video_processor import GreenScreenAutoEditor
from utils.file_utils import get_supported_files, get_matched_file_pairs, validate_file_pairs

class ProcessingPanel:
    def __init__(self, parent, template_path, primary_folder, secondary_folder, output_folder, 
                 text_enabled_callback, get_text_config_callback):
        self.parent = parent
        self.template_path = template_path
        self.primary_folder = primary_folder
        self.secondary_folder = secondary_folder
        self.output_folder = output_folder
        self.text_enabled_callback = text_enabled_callback
        self.get_text_config_callback = get_text_config_callback
        
        # Processing state
        self.processing = False
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup processing section"""
        self.frame = ttk.LabelFrame(self.parent, text="Processing", padding="15")
        self.frame.columnconfigure(0, weight=1)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.frame, mode='indeterminate', length=400)
        self.progress.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Status label
        self.status_label = ttk.Label(self.frame, text="Ready to process", font=('Arial', 10))
        self.status_label.grid(row=1, column=0, sticky=tk.W, pady=(0, 15))
        
        # Button frame
        button_frame = ttk.Frame(self.frame)
        button_frame.grid(row=2, column=0, pady=(0, 10))
        
        # Process button
        self.process_button = ttk.Button(button_frame, text="Start Processing", command=self.start_processing)
        self.process_button.grid(row=0, column=0, padx=(0, 10))
        
        # Stop button
        self.stop_button = ttk.Button(button_frame, text="Stop Processing", command=self.stop_processing, state='disabled')
        self.stop_button.grid(row=0, column=1)
        
        # Info label
        info_frame = ttk.Frame(self.frame)
        info_frame.grid(row=3, column=0, pady=(10, 0), sticky=(tk.W, tk.E))
        
        info_label = ttk.Label(info_frame, text="✓ HD Quality Output  ✓ Auto Duration Sync  ✓ Mixed Media Support", 
                              font=('Arial', 8), foreground='gray')
        info_label.grid(row=0, column=0, sticky=tk.W)
        
        # Quality info
        quality_label = ttk.Label(info_frame, text="Output: 1080p+ with high bitrate encoding", 
                                 font=('Arial', 8), foreground='blue')
        quality_label.grid(row=1, column=0, sticky=tk.W, pady=(2, 0))
        
        # Performance info
        perf_label = ttk.Label(info_frame, text="🚀 Multi-threading + GPU acceleration enabled", 
                              font=('Arial', 8), foreground='green')
        perf_label.grid(row=2, column=0, sticky=tk.W, pady=(2, 0))
    
    def start_processing(self):
        """Start processing videos"""
        if not self.validate_inputs():
            return
        
        self.processing = True
        self.process_button.configure(state='disabled')
        self.stop_button.configure(state='normal')
        self.progress.start()
        self.status_label.configure(text="Processing...")
        
        # Create configuration
        config = self.get_text_config_callback()
        
        # Start processing in separate thread
        self.processing_thread = threading.Thread(target=self.process_videos, args=(config,))
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def stop_processing(self):
        """Stop processing"""
        self.processing = False
        self.process_button.configure(state='normal')
        self.stop_button.configure(state='disabled')
        self.progress.stop()
        self.status_label.configure(text="Processing stopped")
    
    def process_videos(self, config):
        """Process videos in separate thread"""
        try:
            # Get matched file pairs
            if self.secondary_folder.get():
                # Use file matching when secondary folder is provided
                file_pairs = get_matched_file_pairs(self.primary_folder.get(), self.secondary_folder.get())
                
                if not file_pairs:
                    raise ValueError("No matching file pairs found between primary and secondary folders")
                
                # Validate file pairs
                valid_pairs = validate_file_pairs(file_pairs)
                
                if not valid_pairs:
                    raise ValueError("No valid file pairs found")
                
                if len(valid_pairs) < len(file_pairs):
                    print(f"⚠️ Processing {len(valid_pairs)} valid pairs out of {len(file_pairs)} total pairs")
                
            else:
                # No secondary folder - process primary files only
                primary_files = get_supported_files(self.primary_folder.get())
                
                if not primary_files:
                    raise ValueError("No supported files found in primary folder")
                
                # Create pairs with None for secondary
                valid_pairs = [(primary_file, None) for primary_file in primary_files]
            
            print(f"\n🎬 Processing {len(valid_pairs)} file pairs...")
            
            # Process each file pair
            for i, (primary_file, secondary_file) in enumerate(valid_pairs):
                if not self.processing:
                    break
                
                # Get output filename
                output_name = f"output_{Path(primary_file).stem}.mp4"
                output_path = os.path.join(self.output_folder.get(), output_name)
                
                # Update status on main thread
                def update_status(filename, pair_num, total_pairs):
                    self.status_label.configure(text=f"Processing {filename} ({pair_num}/{total_pairs})...")
                
                self.parent.after(0, lambda f=Path(primary_file).name, n=i+1, t=len(valid_pairs): update_status(f, n, t))
                
                # Create processor
                processor = GreenScreenAutoEditor(
                    template_path=self.template_path.get(),
                    video1_path=primary_file,
                    video2_path=secondary_file,
                    output_path=output_path,
                    text_overlay_config=config if self.text_enabled_callback() else None,
                    filename_for_text=Path(primary_file).stem
                )
                
                # Process
                processor.process_video()
                
                print(f"✅ Completed pair {i+1}/{len(valid_pairs)}: {Path(primary_file).name}")
                print("-" * 50)
            
            # Finished
            self.parent.after(0, self.processing_finished)
            
        except Exception as e:
            error_msg = str(e)
            self.parent.after(0, lambda msg=error_msg: self.processing_error(msg))
    
    def processing_finished(self):
        """Called when processing is finished"""
        self.processing = False
        self.process_button.configure(state='normal')
        self.stop_button.configure(state='disabled')
        self.progress.stop()
        self.status_label.configure(text="Processing completed successfully!")
        messagebox.showinfo("Success", "Video processing completed successfully!")
    
    def processing_error(self, error_msg):
        """Called when processing error occurs"""
        self.processing = False
        self.process_button.configure(state='normal')
        self.stop_button.configure(state='disabled')
        self.progress.stop()
        self.status_label.configure(text=f"Error: {error_msg}")
        messagebox.showerror("Error", f"Processing failed: {error_msg}")
    
    def validate_inputs(self):
        """Validate user inputs"""
        if not self.template_path.get():
            messagebox.showwarning("Warning", "Please select a template file")
            return False
        
        if not self.primary_folder.get():
            messagebox.showwarning("Warning", "Please select a primary folder")
            return False
        
        if not self.output_folder.get():
            messagebox.showwarning("Warning", "Please select an output folder")
            return False
        
        if not os.path.exists(self.template_path.get()):
            messagebox.showerror("Error", "Template file does not exist")
            return False
        
        if not os.path.exists(self.primary_folder.get()):
            messagebox.showerror("Error", "Primary folder does not exist")
            return False
        
        if not os.path.exists(self.output_folder.get()):
            messagebox.showerror("Error", "Output folder does not exist")
            return False
        
        return True
    
    def grid(self, **kwargs):
        """Grid the frame"""
        self.frame.grid(**kwargs)