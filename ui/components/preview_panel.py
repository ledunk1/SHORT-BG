import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from core.text_overlay import TextOverlayRenderer

class PreviewPanel:
    def __init__(self, parent, template_path, text_enabled_callback, get_text_config_callback):
        self.parent = parent
        self.template_path = template_path
        self.text_enabled_callback = text_enabled_callback
        self.get_text_config_callback = get_text_config_callback
        self.preview_image = None
        self.text_renderer = None
        self.on_canvas_click = None  # Will be set by main window
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup preview section"""
        self.frame = ttk.LabelFrame(self.parent, text="Preview", padding="10")
        self.frame.columnconfigure(0, weight=1)
        
        # Preview canvas - larger size for better visibility
        self.preview_canvas = tk.Canvas(self.frame, width=500, height=400, bg='lightgray')
        self.preview_canvas.grid(row=0, column=0, pady=(0, 10))
        
        # Bind canvas click for manual positioning
        self.preview_canvas.bind('<Button-1>', self._on_canvas_click)
        
        # Preview controls
        preview_controls = ttk.Frame(self.frame)
        preview_controls.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        ttk.Button(preview_controls, text="Refresh Preview", command=self.refresh_preview).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(preview_controls, text="Full Preview", command=self.open_full_preview).grid(row=0, column=1)
        
        # Info label
        self.info_label = ttk.Label(preview_controls, text="Click on preview to set text position", font=('Arial', 8))
        self.info_label.grid(row=0, column=2, padx=(20, 0))
    
    def _on_canvas_click(self, event):
        """Internal canvas click handler"""
        if self.on_canvas_click:
            self.on_canvas_click(event)
    
    def refresh_preview(self):
        """Refresh the preview canvas"""
        if not self.template_path.get():
            return
        
        try:
            # Load and display first frame of template
            if self.template_path.get().lower().endswith(('.png', '.jpg', '.jpeg')):
                # Static image
                image = Image.open(self.template_path.get())
            else:
                # Video/GIF - get first frame
                cap = cv2.VideoCapture(self.template_path.get())
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame)
                else:
                    return
                cap.release()
            
            # Store original image for text overlay
            self.preview_image = image.copy()
            
            # Apply text overlay if enabled
            if self.text_enabled_callback():
                image = self.apply_text_overlay_to_image(image)
            
            # Resize to fit canvas while maintaining aspect ratio
            display_image = image.copy()
            display_image.thumbnail((500, 400), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(display_image)
            
            # Clear and draw
            self.preview_canvas.delete("all")
            
            # Center the image in canvas
            canvas_width = 500
            canvas_height = 400
            img_width = display_image.width
            img_height = display_image.height
            
            x = (canvas_width - img_width) // 2
            y = (canvas_height - img_height) // 2
            
            self.preview_canvas.create_image(x + img_width//2, y + img_height//2, image=photo)
            
            # Keep reference to prevent garbage collection
            self.preview_canvas.image = photo
            
        except Exception as e:
            print(f"Preview error: {e}")
    
    def open_full_preview(self):
        """Open full preview window"""
        if not self.template_path.get():
            from tkinter import messagebox
            messagebox.showwarning("Warning", "Please select a template file first")
            return
        
        try:
            from ui.preview_window import PreviewWindow
            preview_window = PreviewWindow(self.parent, self.template_path.get())
        except Exception as e:
            from tkinter import messagebox
            messagebox.showerror("Error", f"Failed to preview template: {str(e)}")
    
    def apply_text_overlay_to_image(self, image):
        """Apply text overlay to PIL image for preview"""
        try:
            # Convert PIL to OpenCV
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Create text renderer with current config
            config = self.get_text_config_callback()
            if not self.text_renderer:
                self.text_renderer = TextOverlayRenderer(config)
            else:
                self.text_renderer.update_config(config)
            
            # Apply text overlay
            cv_image_with_text = self.text_renderer.render_text(cv_image, config.preview_text)
            
            # Convert back to PIL
            image_with_text = Image.fromarray(cv2.cvtColor(cv_image_with_text, cv2.COLOR_BGR2RGB))
            
            return image_with_text
            
        except Exception as e:
            print(f"Text overlay preview error: {e}")
            return image
    
    def get_preview_image(self):
        """Get the current preview image"""
        return self.preview_image
    
    def grid(self, **kwargs):
        """Grid the frame"""
        self.frame.grid(**kwargs)