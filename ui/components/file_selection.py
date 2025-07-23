import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
from pathlib import Path
from utils.file_utils import get_matched_file_pairs

class FileSelectionFrame:
    def __init__(self, parent, template_path, primary_folder, secondary_folder, output_folder, preview_callback):
        self.parent = parent
        self.template_path = template_path
        self.primary_folder = primary_folder
        self.secondary_folder = secondary_folder
        self.output_folder = output_folder
        self.preview_callback = preview_callback
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup file selection UI"""
        # Files section
        self.frame = ttk.LabelFrame(self.parent, text="File Selection", padding="10")
        self.frame.columnconfigure(1, weight=1)
        
        # Template selection
        ttk.Label(self.frame, text="Template:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        template_frame = ttk.Frame(self.frame)
        template_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=(0, 5))
        template_frame.columnconfigure(0, weight=1)
        
        ttk.Entry(template_frame, textvariable=self.template_path).grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        ttk.Button(template_frame, text="Browse", command=self.browse_template).grid(row=0, column=1)
        ttk.Button(template_frame, text="Preview", command=self.preview_template).grid(row=0, column=2, padx=(5, 0))
        
        # Primary folder selection
        ttk.Label(self.frame, text="Primary Folder:").grid(row=1, column=0, sticky=tk.W, pady=(0, 5))
        primary_frame = ttk.Frame(self.frame)
        primary_frame.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=(0, 5))
        primary_frame.columnconfigure(0, weight=1)
        
        ttk.Entry(primary_frame, textvariable=self.primary_folder).grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        ttk.Button(primary_frame, text="Browse", command=self.browse_primary_folder).grid(row=0, column=1)
        
        # Secondary folder selection
        ttk.Label(self.frame, text="Secondary Folder:").grid(row=2, column=0, sticky=tk.W, pady=(0, 5))
        secondary_frame = ttk.Frame(self.frame)
        secondary_frame.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=(0, 5))
        secondary_frame.columnconfigure(0, weight=1)
        
        ttk.Entry(secondary_frame, textvariable=self.secondary_folder).grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        ttk.Button(secondary_frame, text="Browse", command=self.browse_secondary_folder).grid(row=0, column=1)
        ttk.Button(secondary_frame, text="Check Match", command=self.check_file_matching).grid(row=0, column=2, padx=(5, 0))
        
        # Output folder selection
        ttk.Label(self.frame, text="Output Folder:").grid(row=3, column=0, sticky=tk.W, pady=(0, 5))
        output_frame = ttk.Frame(self.frame)
        output_frame.grid(row=3, column=1, sticky=(tk.W, tk.E), pady=(0, 5))
        output_frame.columnconfigure(0, weight=1)
        
        ttk.Entry(output_frame, textvariable=self.output_folder).grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        ttk.Button(output_frame, text="Browse", command=self.browse_output_folder).grid(row=0, column=1)
    
    def browse_template(self):
        """Browse for template file"""
        file_path = filedialog.askopenfilename(
            title="Select Template File",
            filetypes=[
                ("All supported", "*.mp4;*.avi;*.mov;*.mkv;*.gif;*.png;*.jpg;*.jpeg"),
                ("Video files", "*.mp4;*.avi;*.mov;*.mkv"),
                ("Image files", "*.png;*.jpg;*.jpeg;*.gif"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.template_path.set(file_path)
            self.preview_callback()
    
    def browse_primary_folder(self):
        """Browse for primary folder"""
        folder_path = filedialog.askdirectory(title="Select Primary Folder")
        if folder_path:
            self.primary_folder.set(folder_path)
            
    def browse_secondary_folder(self):
        """Browse for secondary folder"""
        folder_path = filedialog.askdirectory(title="Select Secondary Folder")
        if folder_path:
            self.secondary_folder.set(folder_path)
            # Auto-check matching when secondary folder is selected
            self.check_file_matching()
            
    def browse_output_folder(self):
        """Browse for output folder"""
        folder_path = filedialog.askdirectory(title="Select Output Folder")
        if folder_path:
            self.output_folder.set(folder_path)
    
    def preview_template(self):
        """Preview template in separate window"""
        if not self.template_path.get():
            messagebox.showwarning("Warning", "Please select a template file first")
            return
        
        try:
            from ui.preview_window import PreviewWindow
            preview_window = PreviewWindow(self.parent, self.template_path.get())
        except Exception as e:
            messagebox.showerror("Error", f"Failed to preview template: {str(e)}")
    
    def check_file_matching(self):
        """Check file matching between primary and secondary folders"""
        if not self.primary_folder.get() or not self.secondary_folder.get():
            return
        
        try:
            # Get matched file pairs
            file_pairs = get_matched_file_pairs(self.primary_folder.get(), self.secondary_folder.get())
            
            if file_pairs:
                # Show matching results
                match_info = f"✅ Found {len(file_pairs)} matching file pairs:\n\n"
                
                for i, (primary, secondary) in enumerate(file_pairs[:10], 1):  # Show first 10
                    match_info += f"{i}. {Path(primary).name} ↔ {Path(secondary).name}\n"
                
                if len(file_pairs) > 10:
                    match_info += f"\n... and {len(file_pairs) - 10} more pairs"
                
                messagebox.showinfo("File Matching Results", match_info)
            else:
                messagebox.showwarning("No Matches Found", 
                                     "No matching files found between primary and secondary folders.\n\n"
                                     "Make sure files have the same base name:\n"
                                     "• Primary: 'video1.mp4'\n"
                                     "• Secondary: 'video1.mov' ✅\n"
                                     "• Secondary: 'different.mp4' ❌")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to check file matching: {str(e)}")
    
    def grid(self, **kwargs):
        """Grid the frame"""
        self.frame.grid(**kwargs)