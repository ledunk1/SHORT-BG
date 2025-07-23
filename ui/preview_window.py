import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import threading
import time

class PreviewWindow:
    def __init__(self, parent, template_path):
        self.parent = parent
        self.template_path = template_path
        self.playing = False
        self.frames = []
        self.current_frame = 0
        self.fps = 30
        
        # Create window
        self.window = tk.Toplevel(parent)
        self.window.title("Template Preview")
        self.window.geometry("600x500")
        
        # Load template
        self.load_template()
        
        # Setup UI
        self.setup_ui()
        
        # Start preview if it's a video/gif
        if len(self.frames) > 1:
            self.start_preview()
    
    def load_template(self):
        """Load template frames"""
        try:
            if self.template_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Static image
                image = Image.open(self.template_path)
                self.frames = [image]
                self.fps = 1
            else:
                # Video/GIF
                cap = cv2.VideoCapture(self.template_path)
                self.fps = cap.get(cv2.CAP_PROP_FPS) or 30
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame)
                    self.frames.append(image)
                
                cap.release()
                
            if not self.frames:
                raise ValueError("No frames loaded")
                
        except Exception as e:
            raise Exception(f"Failed to load template: {str(e)}")
    
    def setup_ui(self):
        """Setup UI components"""
        # Main frame
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Preview canvas
        self.canvas = tk.Canvas(main_frame, width=500, height=400, bg='black')
        self.canvas.pack(pady=(0, 10))
        
        # Control frame
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X)
        
        # Controls for video/gif
        if len(self.frames) > 1:
            self.play_button = ttk.Button(control_frame, text="Play", command=self.toggle_play)
            self.play_button.pack(side=tk.LEFT, padx=(0, 5))
            
            # Frame slider
            self.frame_var = tk.IntVar(value=0)
            self.frame_slider = ttk.Scale(control_frame, from_=0, to=len(self.frames)-1, 
                                         orient=tk.HORIZONTAL, variable=self.frame_var,
                                         command=self.on_frame_change)
            self.frame_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
            
            # Frame counter
            self.frame_label = ttk.Label(control_frame, text=f"Frame: 1/{len(self.frames)}")
            self.frame_label.pack(side=tk.RIGHT)
        
        # Show first frame
        self.show_frame(0)
    
    def show_frame(self, frame_index):
        """Show specific frame"""
        if 0 <= frame_index < len(self.frames):
            # Get frame
            frame = self.frames[frame_index].copy()
            
            # Resize to fit canvas
            frame.thumbnail((500, 400))
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(frame)
            
            # Clear canvas and show frame
            self.canvas.delete("all")
            self.canvas.create_image(250, 200, image=photo)
            
            # Keep reference
            self.canvas.image = photo
            
            # Update frame counter
            if len(self.frames) > 1:
                self.frame_label.configure(text=f"Frame: {frame_index + 1}/{len(self.frames)}")
    
    def toggle_play(self):
        """Toggle play/pause"""
        if self.playing:
            self.playing = False
            self.play_button.configure(text="Play")
        else:
            self.playing = True
            self.play_button.configure(text="Pause")
            if not hasattr(self, 'preview_thread') or not self.preview_thread.is_alive():
                self.start_preview()
    
    def start_preview(self):
        """Start preview animation"""
        if len(self.frames) > 1:
            self.playing = True
            if hasattr(self, 'play_button'):
                self.play_button.configure(text="Pause")
            self.preview_thread = threading.Thread(target=self.preview_loop)
            self.preview_thread.daemon = True
            self.preview_thread.start()
    
    def preview_loop(self):
        """Preview animation loop"""
        while self.playing and self.window.winfo_exists():
            try:
                # Update frame
                self.current_frame = (self.current_frame + 1) % len(self.frames)
                
                # Update UI on main thread
                self.window.after(0, self.update_frame)
                
                # Wait for next frame
                time.sleep(1.0 / self.fps)
                
            except tk.TclError:
                # Window closed
                break
    
    def update_frame(self):
        """Update frame on main thread"""
        if self.window.winfo_exists():
            self.show_frame(self.current_frame)
            self.frame_var.set(self.current_frame)
    
    def on_frame_change(self, value):
        """Handle frame slider change"""
        if not self.playing:
            frame_index = int(float(value))
            self.current_frame = frame_index
            self.show_frame(frame_index)