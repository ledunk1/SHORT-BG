import tkinter as tk
from tkinter import ttk, colorchooser
from core.text_overlay import TextOverlayConfig

class TextOverlayPanel:
    def __init__(self, parent, text_change_callback):
        self.parent = parent
        self.text_change_callback = text_change_callback
        
        # Text overlay variables
        self.text_enabled = tk.BooleanVar(value=True)
        self.preview_text_var = tk.StringVar(value="Sample Text")
        self.font_var = tk.StringVar(value="Arial")
        self.font_size_var = tk.IntVar(value=24)
        self.color_var = tk.StringVar(value="#FFFFFF")
        self.position_var = tk.StringVar(value="bottom-center")
        self.manual_position_enabled = tk.BooleanVar(value=False)
        self.manual_x_var = tk.IntVar(value=0)
        self.manual_y_var = tk.IntVar(value=0)
        self.outline_enabled = tk.BooleanVar(value=True)
        self.autowrap_enabled = tk.BooleanVar(value=True)
        self.emoji_support = tk.BooleanVar(value=True)
        self.emoji_size_var = tk.IntVar(value=0)  # 0 = auto-size
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup text overlay configuration"""
        self.frame = ttk.LabelFrame(self.parent, text="Text Overlay Settings", padding="10")
        self.frame.columnconfigure(0, weight=1)
        
        # Enable/disable text overlay
        ttk.Checkbutton(self.frame, text="Enable Text Overlay", variable=self.text_enabled,
                       command=self.toggle_text_overlay).grid(row=0, column=0, sticky=tk.W, pady=(0, 10))
        
        # Text settings frame
        self.text_settings_frame = ttk.Frame(self.frame)
        self.text_settings_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        self.text_settings_frame.columnconfigure(0, weight=1)
        
        # Preview text input
        text_frame = ttk.LabelFrame(self.text_settings_frame, text="Preview Text", padding="5")
        text_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        text_frame.columnconfigure(0, weight=1)
        
        preview_text_entry = ttk.Entry(text_frame, textvariable=self.preview_text_var)
        preview_text_entry.grid(row=0, column=0, sticky=(tk.W, tk.E))
        preview_text_entry.bind('<KeyRelease>', self.on_text_change)
        
        # Add emoji support info
        emoji_info = ttk.Label(text_frame, text="💡 Tip: Place emoji PNG files in 'emoji/' folder (e.g., 1F600.png for 😀)", 
                              font=('Arial', 8), foreground='gray')
        emoji_info.grid(row=1, column=0, sticky=tk.W, pady=(2, 0))
        
        # Font settings
        font_frame = ttk.LabelFrame(self.text_settings_frame, text="Font Settings", padding="5")
        font_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        font_frame.columnconfigure(1, weight=1)
        
        # Font family
        ttk.Label(font_frame, text="Font:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        font_combo = ttk.Combobox(font_frame, textvariable=self.font_var, width=15)
        font_combo['values'] = ('Arial', 'Helvetica', 'Times', 'Courier', 'Verdana', 'Georgia', 'Impact', 'Comic Sans MS')
        font_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=(0, 5), padx=(5, 0))
        font_combo.bind('<<ComboboxSelected>>', self.on_text_change)
        
        # Font size
        ttk.Label(font_frame, text="Size:").grid(row=1, column=0, sticky=tk.W, pady=(0, 5))
        font_size_spin = ttk.Spinbox(font_frame, from_=8, to=72, textvariable=self.font_size_var, width=10)
        font_size_spin.grid(row=1, column=1, sticky=tk.W, pady=(0, 5), padx=(5, 0))
        font_size_spin.bind('<KeyRelease>', self.on_text_change)
        font_size_spin.bind('<<Increment>>', self.on_text_change)
        font_size_spin.bind('<<Decrement>>', self.on_text_change)
        
        # Color settings
        color_frame = ttk.LabelFrame(self.text_settings_frame, text="Color Settings", padding="5")
        color_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(color_frame, text="Text Color:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        color_button_frame = ttk.Frame(color_frame)
        color_button_frame.grid(row=0, column=1, sticky=tk.W, pady=(0, 5), padx=(5, 0))
        
        self.color_sample = tk.Frame(color_button_frame, width=30, height=20, bg=self.color_var.get())
        self.color_sample.grid(row=0, column=0, padx=(0, 5))
        ttk.Button(color_button_frame, text="Choose", command=self.choose_color).grid(row=0, column=1)
        
        # Position settings
        position_frame = ttk.LabelFrame(self.text_settings_frame, text="Position Settings", padding="5")
        position_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        position_frame.columnconfigure(1, weight=1)
        
        # Preset position
        ttk.Label(position_frame, text="Position:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        self.position_combo = ttk.Combobox(position_frame, textvariable=self.position_var)
        self.position_combo['values'] = ('top-left', 'top-center', 'top-right', 'center-left', 'center', 'center-right', 
                                        'bottom-left', 'bottom-center', 'bottom-right')
        self.position_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=(0, 5), padx=(5, 0))
        self.position_combo.bind('<<ComboboxSelected>>', self.on_text_change)
        
        # Manual position toggle
        ttk.Checkbutton(position_frame, text="Manual Position (Click on preview to set)", 
                       variable=self.manual_position_enabled,
                       command=self.toggle_manual_position).grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(10, 5))
        
        # Manual position controls
        manual_frame = ttk.Frame(position_frame)
        manual_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))
        manual_frame.columnconfigure(1, weight=1)
        manual_frame.columnconfigure(3, weight=1)
        
        # X position
        ttk.Label(manual_frame, text="X:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        self.manual_x_scale = ttk.Scale(manual_frame, from_=0, to=1920, orient=tk.HORIZONTAL, 
                                       variable=self.manual_x_var, command=self.on_manual_position_change)
        self.manual_x_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(5, 5), pady=(0, 5))
        self.manual_x_scale.configure(state='disabled')
        
        self.manual_x_entry = ttk.Entry(manual_frame, textvariable=self.manual_x_var, width=8, state='disabled')
        self.manual_x_entry.grid(row=0, column=2, padx=(0, 10), pady=(0, 5))
        self.manual_x_entry.bind('<KeyRelease>', self.on_text_change)
        
        # Y position
        ttk.Label(manual_frame, text="Y:").grid(row=1, column=0, sticky=tk.W, pady=(0, 5))
        self.manual_y_scale = ttk.Scale(manual_frame, from_=0, to=1080, orient=tk.HORIZONTAL, 
                                       variable=self.manual_y_var, command=self.on_manual_position_change)
        self.manual_y_scale.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(5, 5), pady=(0, 5))
        self.manual_y_scale.configure(state='disabled')
        
        self.manual_y_entry = ttk.Entry(manual_frame, textvariable=self.manual_y_var, width=8, state='disabled')
        self.manual_y_entry.grid(row=1, column=2, pady=(0, 5))
        self.manual_y_entry.bind('<KeyRelease>', self.on_text_change)
        
        # Additional settings
        additional_frame = ttk.LabelFrame(self.text_settings_frame, text="Additional Settings", padding="5")
        additional_frame.grid(row=4, column=0, sticky=(tk.W, tk.E))
        
        # Text outline
        ttk.Checkbutton(additional_frame, text="Enable Outline", variable=self.outline_enabled, 
                       command=self.on_text_change).grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        # Auto-wrap
        ttk.Checkbutton(additional_frame, text="Enable Auto-wrap", variable=self.autowrap_enabled, 
                       command=self.on_text_change).grid(row=1, column=0, sticky=tk.W)
        
        # Emoji support
        ttk.Checkbutton(additional_frame, text="Enable Emoji Support 😀", variable=self.emoji_support, 
                       command=self.on_text_change).grid(row=2, column=0, sticky=tk.W, pady=(5, 0))
        
        # Emoji size
        emoji_size_frame = ttk.Frame(additional_frame)
        emoji_size_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        ttk.Label(emoji_size_frame, text="Emoji Size:").grid(row=0, column=0, sticky=tk.W)
        emoji_size_spin = ttk.Spinbox(emoji_size_frame, from_=0, to=100, textvariable=self.emoji_size_var, width=10)
        emoji_size_spin.grid(row=0, column=1, sticky=tk.W, padx=(5, 0))
        emoji_size_spin.bind('<KeyRelease>', self.on_text_change)
        emoji_size_spin.bind('<<Increment>>', self.on_text_change)
        emoji_size_spin.bind('<<Decrement>>', self.on_text_change)
        
        ttk.Label(emoji_size_frame, text="(0 = auto-size)", font=('Arial', 8), foreground='gray').grid(row=0, column=2, padx=(5, 0))
    
    def toggle_text_overlay(self):
        """Toggle text overlay settings"""
        state = 'normal' if self.text_enabled.get() else 'disabled'
        
        # Enable/disable all child widgets
        self.set_frame_state(self.text_settings_frame, state)
        
        # Handle manual position controls separately
        self.toggle_manual_position()
        
        self.text_change_callback()
    
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
    
    def toggle_manual_position(self):
        """Toggle manual position controls"""
        if self.manual_position_enabled.get() and self.text_enabled.get():
            # Enable manual position controls
            self.manual_x_scale.configure(state='normal')
            self.manual_y_scale.configure(state='normal')
            self.manual_x_entry.configure(state='normal')
            self.manual_y_entry.configure(state='normal')
            # Disable preset position combo
            self.position_combo.configure(state='disabled')
        else:
            # Disable manual position controls
            self.manual_x_scale.configure(state='disabled')
            self.manual_y_scale.configure(state='disabled')
            self.manual_x_entry.configure(state='disabled')
            self.manual_y_entry.configure(state='disabled')
            # Enable preset position combo
            self.position_combo.configure(state='normal' if self.text_enabled.get() else 'disabled')
        
        self.text_change_callback()
    
    def choose_color(self):
        """Choose text color"""
        color = colorchooser.askcolor(color=self.color_var.get())
        if color[1]:  # If user didn't cancel
            self.color_var.set(color[1])
            self.color_sample.configure(bg=color[1])
            self.text_change_callback()
    
    def on_text_change(self, event=None):
        """Handle text overlay setting changes"""
        self.text_change_callback()
    
    def on_manual_position_change(self, value):
        """Handle manual position slider changes"""
        if self.manual_position_enabled.get():
            self.text_change_callback()
    
    def update_manual_position_ranges(self, width, height):
        """Update manual position scale ranges"""
        self.manual_x_scale.configure(to=width)
        self.manual_y_scale.configure(to=height)
    
    def get_text_overlay_config(self):
        """Get text overlay configuration"""
        config = TextOverlayConfig()
        config.font_family = self.font_var.get()
        config.font_size = self.font_size_var.get()
        config.color = self.color_var.get()
        config.position = self.position_var.get()
        config.use_manual_position = self.manual_position_enabled.get()
        config.manual_x = self.manual_x_var.get()
        config.manual_y = self.manual_y_var.get()
        config.outline_enabled = self.outline_enabled.get()
        config.autowrap_enabled = self.autowrap_enabled.get()
        config.emoji_support = self.emoji_support.get()
        config.preview_text = self.preview_text_var.get()
        config.emoji_size = self.emoji_size_var.get() if self.emoji_size_var.get() > 0 else None
        return config
    
    def grid(self, **kwargs):
        """Grid the frame"""
        self.frame.grid(**kwargs)