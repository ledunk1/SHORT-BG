import tkinter as tk
from tkinter import ttk
import os
from pathlib import Path

from core.text_overlay import TextOverlayConfig
from ui.components.file_selection import FileSelectionFrame
from ui.components.preview_panel import PreviewPanel
from ui.components.text_overlay_panel import TextOverlayPanel
from ui.components.processing_panel import ProcessingPanel

class MainWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Green Screen Auto Editor")
        
        # Get screen dimensions and set window size
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        
        # Set window size to 90% of screen size
        window_width = int(screen_width * 0.9)
        window_height = int(screen_height * 0.9)
        
        # Center window on screen
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.root.minsize(1000, 700)  # Minimum size
        
        # Initialize variables
        self.template_path = tk.StringVar()
        self.primary_folder = tk.StringVar()
        self.secondary_folder = tk.StringVar()
        self.output_folder = tk.StringVar()
        
        # Setup UI
        self.setup_ui()
        
        # Load settings
        self.load_settings()
        
    def setup_ui(self):
        """Setup the main UI components"""
        # Create main scrollable frame
        self.setup_scrollable_frame()
        
        # Setup components
        self.setup_components()
        
    def setup_scrollable_frame(self):
        """Setup scrollable main frame"""
        # Create main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill="both", expand=True)
        
        # Create canvas and scrollbar
        self.canvas = tk.Canvas(main_container)
        self.scrollbar = ttk.Scrollbar(main_container, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        # Configure scrolling
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        # Create window in canvas
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Pack canvas and scrollbar
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # Bind events
        self.canvas.bind('<Configure>', self.on_canvas_configure)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
        # Main content frame with padding
        self.main_frame = ttk.Frame(self.scrollable_frame, padding="15")
        self.main_frame.pack(fill="both", expand=True)
        self.main_frame.columnconfigure(0, weight=1)
    
    def on_canvas_configure(self, event):
        """Handle canvas resize"""
        # Update scroll region
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
        # Make scrollable frame fill canvas width
        canvas_width = event.width
        self.canvas.itemconfig(self.canvas_window, width=canvas_width)
    
    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling"""
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    def setup_components(self):
        """Setup UI components"""
        # File selection section
        self.file_selection = FileSelectionFrame(
            self.main_frame, 
            self.template_path, 
            self.primary_folder, 
            self.secondary_folder, 
            self.output_folder,
            self.refresh_preview
        )
        self.file_selection.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N), pady=(0, 15))
        
        # Create two-column layout for preview and text overlay
        content_frame = ttk.Frame(self.main_frame)
        content_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N), pady=(0, 15))
        content_frame.columnconfigure(0, weight=1)
        content_frame.columnconfigure(1, weight=1)
        
        # Preview section (left column)
        self.preview_panel = PreviewPanel(
            content_frame,
            self.template_path,
            self.get_text_enabled,
            self.get_text_overlay_config
        )
        self.preview_panel.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N), padx=(0, 10))
        
        # Connect canvas click handler
        self.preview_panel.on_canvas_click = self.on_canvas_click
        
        # Text overlay section (right column)
        self.text_overlay_panel = TextOverlayPanel(
            content_frame,
            self.on_text_change
        )
        self.text_overlay_panel.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N), padx=(10, 0))
        
        # Processing section (full width)
        self.processing_panel = ProcessingPanel(
            self.main_frame,
            self.template_path,
            self.primary_folder,
            self.secondary_folder,
            self.output_folder,
            self.get_text_enabled,
            self.get_text_overlay_config
        )
        self.processing_panel.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N), pady=(15, 0))
    
    def get_text_enabled(self):
        """Get text enabled status"""
        return self.text_overlay_panel.text_enabled.get()
    
    def refresh_preview(self):
        """Refresh the preview"""
        self.preview_panel.refresh_preview()
        
        # Update manual position scale ranges based on image size
        if self.preview_panel.preview_image:
            self.text_overlay_panel.update_manual_position_ranges(
                self.preview_panel.preview_image.width,
                self.preview_panel.preview_image.height
            )
    
    def on_text_change(self):
        """Handle text overlay setting changes"""
        self.refresh_preview()
    
    def on_canvas_click(self, event):
        """Handle canvas click for manual positioning"""
        if (self.text_overlay_panel.manual_position_enabled.get() and 
            self.text_overlay_panel.text_enabled.get() and 
            self.preview_panel.preview_image):
            
            # Calculate actual position in original image
            canvas_width = self.preview_panel.preview_canvas.winfo_width()
            canvas_height = self.preview_panel.preview_canvas.winfo_height()
            
            # Get image dimensions after thumbnail
            img_copy = self.preview_panel.preview_image.copy()
            img_copy.thumbnail((500, 400))
            
            # Calculate scale factors
            scale_x = self.preview_panel.preview_image.width / img_copy.width
            scale_y = self.preview_panel.preview_image.height / img_copy.height
            
            # Calculate offset (image is centered in canvas)
            offset_x = (canvas_width - img_copy.width) // 2
            offset_y = (canvas_height - img_copy.height) // 2
            
            # Convert click position to image coordinates
            img_x = int((event.x - offset_x) * scale_x)
            img_y = int((event.y - offset_y) * scale_y)
            
            # Clamp to image bounds
            img_x = max(0, min(img_x, self.preview_panel.preview_image.width))
            img_y = max(0, min(img_y, self.preview_panel.preview_image.height))
            
            # Update manual position
            self.text_overlay_panel.manual_x_var.set(img_x)
            self.text_overlay_panel.manual_y_var.set(img_y)
            
            # Refresh preview
            self.refresh_preview()
    
    def get_text_overlay_config(self):
        """Get text overlay configuration"""
        return self.text_overlay_panel.get_text_overlay_config()
    
    def load_settings(self):
        """Load settings from file"""
        # TODO: Implement settings loading
        pass
    
    def save_settings(self):
        """Save settings to file"""
        # TODO: Implement settings saving
        pass