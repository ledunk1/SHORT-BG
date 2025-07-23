import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import textwrap
from typing import Tuple, List
import os
from pathlib import Path
import re

class TextOverlayConfig:
    """Configuration for text overlay"""
    def __init__(self):
        self.font_family = "Arial"
        self.font_size = 24
        self.color = "#FFFFFF"  # White
        self.position = "bottom-center"
        self.use_manual_position = False  # Whether to use manual X,Y positioning
        self.manual_x = 0  # Manual X position (pixels from left)
        self.manual_y = 0  # Manual Y position (pixels from top)
        self.outline_enabled = True
        self.outline_color = "#000000"  # Black
        self.outline_width = 2
        self.autowrap_enabled = True
        self.max_width_ratio = 0.8  # Maximum text width as ratio of frame width
        self.margin = 20  # Margin from edges
        self.line_spacing = 1.2  # Line spacing multiplier
        self.preview_text = "Sample Text"  # Text for preview
        self.emoji_support = True  # Enable emoji support
        self.emoji_size = None  # Auto-size based on font size if None

class TextOverlayRenderer:
    """Renders text overlay on video frames"""
    
    def __init__(self, config: TextOverlayConfig):
        self.config = config
        self.font = self._load_font()
        self.emoji_folder = "emoji/"
        
    def _load_font(self) -> ImageFont.ImageFont:
        """Load font from system or use default"""
        try:
            # Try to load system font
            font_paths = [
                # Windows
                f"C:/Windows/Fonts/{self.config.font_family.replace(' ', '')}.ttf",
                f"C:/Windows/Fonts/{self.config.font_family.replace(' ', '').lower()}.ttf",
                # macOS
                f"/System/Library/Fonts/{self.config.font_family}.ttf",
                f"/Library/Fonts/{self.config.font_family}.ttf",
                # Linux
                f"/usr/share/fonts/truetype/dejavu/{self.config.font_family}.ttf",
                f"/usr/share/fonts/truetype/liberation/{self.config.font_family}.ttf",
                # Common alternatives
                "/System/Library/Fonts/Arial.ttf",
                "/System/Library/Fonts/Helvetica.ttf",
                "C:/Windows/Fonts/arial.ttf",
                "C:/Windows/Fonts/calibri.ttf",
            ]
            
            for font_path in font_paths:
                if os.path.exists(font_path):
                    return ImageFont.truetype(font_path, self.config.font_size)
            
            # If no system font found, use default
            return ImageFont.load_default()
            
        except Exception as e:
            print(f"Warning: Could not load font {self.config.font_family}, using default: {e}")
            return ImageFont.load_default()
    
    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def _get_emoji_pattern(self):
        """Get regex pattern for emoji detection"""
        return re.compile(
            "["
            "\U0001F600-\U0001F64F" "\U0001F300-\U0001F5FF" "\U0001F680-\U0001F6FF"
            "\U0001F700-\U0001F77F" "\U0001F780-\U0001F7FF" "\U0001F800-\U0001F8FF"
            "\U0001F900-\U0001F9FF" "\U0001FA00-\U0001FA6F" "\U0001FA70-\U0001FAFF"
            "\U00002600-\U000027BF" "\U0001F1E6-\U0001F1FF" "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251" "]+", flags=re.UNICODE,
        )
    
    def _load_emoji(self, emoji_char: str, emoji_size: int) -> Image.Image:
        """Load emoji image from PNG file"""
        try:
            emoji_chars = list(emoji_char)
            emoji_imgs = []
            
            for char in emoji_chars:
                unicode_hex = f"{ord(char):04X}"
                emoji_files = [
                    os.path.join(self.emoji_folder, f"{unicode_hex}.png"),
                    os.path.join(self.emoji_folder, f"{unicode_hex}FE0F.png")
                ]
                
                emoji_file = next((f for f in emoji_files if os.path.exists(f)), None)
                if emoji_file is None and 'FE0F' in unicode_hex:
                    emoji_file = emoji_file.replace("FE0F", "")
                
                if emoji_file and os.path.exists(emoji_file):
                    emoji_img = Image.open(emoji_file).convert("RGBA")
                    emoji_imgs.append(emoji_img.resize((emoji_size, emoji_size), Image.Resampling.LANCZOS))
                else:
                    print(f"Emoji '{char}' tidak ditemukan! (Unicode: {unicode_hex})")
                    return None
            
            if len(emoji_imgs) > 1:
                width, height = emoji_imgs[0].size
                new_img = Image.new("RGBA", (width * len(emoji_imgs), height))
                for i, img in enumerate(emoji_imgs): 
                    new_img.paste(img, (i * width, 0))
                return new_img
            
            return emoji_imgs[0] if emoji_imgs else None
            
        except Exception as e:
            print(f"Error loading emoji {emoji_char}: {e}")
            return None
    
    def _get_text_size(self, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
        """Get text size using PIL"""
        # Create a dummy image to measure text
        dummy_img = Image.new('RGB', (1, 1))
        draw = ImageDraw.Draw(dummy_img)
        
        # Get text bounding box
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            return width, height
        except AttributeError:
            # Fallback for older PIL versions
            width, height = draw.textsize(text, font=font)
            return width, height
    
    def _calculate_content_width(self, parts: List[str], draw: ImageDraw.Draw, 
                                font: ImageFont.ImageFont, emoji_size: int) -> int:
        """Calculate total width of content (text + emoji)"""
        emoji_pattern = self._get_emoji_pattern()
        total_width = 0
        
        for part in parts:
            if emoji_pattern.fullmatch(part):
                total_width += emoji_size
            else:
                text_width, _ = self._get_text_size(part, font)
                total_width += text_width
        
        return total_width
    
    def _smart_text_wrap(self, text: str, draw: ImageDraw.Draw, font: ImageFont.ImageFont, 
                        max_width: int, emoji_size: int) -> List[str]:
        """Smart text wrapping with emoji support"""
        if not self.config.autowrap_enabled:
            return [text]
        
        emoji_pattern = self._get_emoji_pattern()
        
        # Split text into words, preserving emoji
        words = []
        current_word = ""
        
        for char in text:
            if emoji_pattern.match(char):
                if current_word:
                    words.append(current_word)
                    current_word = ""
                words.append(char)
            elif char == ' ':
                if current_word:
                    words.append(current_word)
                    current_word = ""
            else:
                current_word += char
        
        if current_word:
            words.append(current_word)
        
        # Wrap words into lines
        lines = []
        current_line = []
        current_line_width = 0
        
        for word in words:
            # Calculate word width (text or emoji)
            if emoji_pattern.fullmatch(word):
                word_width = emoji_size
            else:
                word_width, _ = self._get_text_size(word, font)
            
            # Add space if not first word in line
            space_width = 0
            if current_line and not emoji_pattern.fullmatch(word):
                space_width, _ = self._get_text_size(" ", font)
            
            # Check if word fits in current line
            if current_line_width + space_width + word_width <= max_width:
                if current_line and not emoji_pattern.fullmatch(word):
                    current_line.append(" ")
                    current_line_width += space_width
                current_line.append(word)
                current_line_width += word_width
            else:
                # Word doesn't fit, create new line
                if current_line:
                    lines.append("".join(current_line))
                current_line = [word]
                current_line_width = word_width
        
        # Add last line
        if current_line:
            lines.append("".join(current_line))
        
        return lines
    
    def _get_text_position(self, text_lines: List[str], font: ImageFont.ImageFont, 
                          frame_width: int, frame_height: int, emoji_size: int) -> Tuple[int, int]:
        """Calculate text position based on configuration"""
        # Use manual position if enabled
        if self.config.use_manual_position:
            return self.config.manual_x, self.config.manual_y
        
        # Create dummy draw for measurements
        dummy_img = Image.new('RGB', (1, 1))
        draw = ImageDraw.Draw(dummy_img)
        
        # Get text dimensions including emoji
        max_line_width = 0
        total_height = 0
        
        emoji_pattern = self._get_emoji_pattern()
        
        for line in text_lines:
            # Split line into parts (text and emoji)
            parts = re.split(f"({emoji_pattern.pattern})", line)
            line_width = self._calculate_content_width(parts, draw, font, emoji_size)
            max_line_width = max(max_line_width, line_width)
            total_height += int(font.size * self.config.line_spacing)
        
        # Calculate position based on config
        if self.config.position.endswith('left'):
            x = self.config.margin
        elif self.config.position.endswith('right'):
            x = frame_width - max_line_width - self.config.margin
        else:  # center
            x = (frame_width - max_line_width) // 2
        
        if self.config.position.startswith('top'):
            y = self.config.margin
        elif self.config.position.startswith('bottom'):
            y = frame_height - total_height - self.config.margin
        else:  # center
            y = (frame_height - total_height) // 2
        
        return x, y
    
    def _render_text_with_emoji_multiline(self, draw: ImageDraw.Draw, lines: List[str], 
                                         font: ImageFont.ImageFont, canvas_width: int, 
                                         canvas_height: int, start_y: int, emoji_size: int, 
                                         text_color: Tuple[int, int, int], 
                                         outline_color: Tuple[int, int, int] = None) -> List[dict]:
        """Render multiple lines of text with emoji support"""
        emoji_pattern = self._get_emoji_pattern()
        rendered_lines = []
        current_y = start_y
        
        # Calculate line height
        line_height = int(font.size * self.config.line_spacing)
        total_height = len(lines) * line_height
        
        # Auto-adjust if exceeds frame
        if current_y + total_height > canvas_height:
            current_y = max(10, canvas_height - total_height - 20)
        
        for line in lines:
            if current_y + line_height > canvas_height:
                break  # Stop if exceeds frame
                
            # Split line into parts (text and emoji)
            parts = re.split(f"({emoji_pattern.pattern})", line)
            
            # Calculate total line width
            total_width = self._calculate_content_width(parts, draw, font, emoji_size)
            
            # Auto-adjust if exceeds frame width
            adjusted_emoji_size = emoji_size
            if total_width > canvas_width - 40:  # 40px margin
                adjusted_emoji_size = min(emoji_size, int(emoji_size * (canvas_width - 40) / total_width))
                total_width = self._calculate_content_width(parts, draw, font, adjusted_emoji_size)
            
            # Position X for center alignment
            x_start = (canvas_width - total_width) // 2
            
            # Render parts
            rendered_items = []
            x_offset = 0
            
            for part in parts:
                if emoji_pattern.fullmatch(part):
                    emoji_img = self._load_emoji(part, adjusted_emoji_size)
                    if emoji_img:
                        rendered_items.append(('emoji', emoji_img, x_offset))
                        x_offset += adjusted_emoji_size
                else:
                    if part.strip():  # Skip empty strings
                        rendered_items.append(('text', part, x_offset))
                        text_width, _ = self._get_text_size(part, font)
                        x_offset += text_width
            
            rendered_lines.append({
                'items': rendered_items,
                'x_start': x_start,
                'y': current_y,
                'emoji_size': adjusted_emoji_size
            })
            
            current_y += line_height
        
        return rendered_lines
    
    def render_text(self, frame: np.ndarray, text: str, frame_index: int = 0) -> np.ndarray:
        """
        Render text overlay on frame with emoji support
        
        Args:
            frame: Input frame in BGR format
            text: Text to render
            frame_index: Frame index (for animations)
            
        Returns:
            Frame with text overlay
        """
        if not text:
            return frame
        
        # Convert frame to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_image)
        
        # Get frame dimensions
        frame_width, frame_height = pil_image.size
        
        # Calculate emoji size (auto-size based on font if not specified)
        emoji_size = self.config.emoji_size if self.config.emoji_size else int(self.config.font_size * 1.2)
        
        # Calculate max text width
        max_text_width = int(frame_width * self.config.max_width_ratio)
        
        # Smart text wrapping with emoji support
        text_lines = self._smart_text_wrap(text, draw, self.font, max_text_width, emoji_size)
        
        # Get text position
        x, y = self._get_text_position(text_lines, self.font, frame_width, frame_height, emoji_size)
        
        # Get colors
        text_color = self._hex_to_rgb(self.config.color)
        outline_color = self._hex_to_rgb(self.config.outline_color)
        
        # Render text with emoji support
        if self.config.emoji_support:
            rendered_lines = self._render_text_with_emoji_multiline(
                draw, text_lines, self.font, frame_width, frame_height, y, 
                emoji_size, text_color, outline_color
            )
            
            # Draw rendered lines
            for line_data in rendered_lines:
                for item_type, content, x_offset in line_data['items']:
                    actual_x = line_data['x_start'] + x_offset
                    actual_y = line_data['y']
                    
                    if item_type == 'emoji':
                        # Draw outline for emoji if enabled
                        if self.config.outline_enabled:
                            for dx in range(-self.config.outline_width, self.config.outline_width + 1):
                                for dy in range(-self.config.outline_width, self.config.outline_width + 1):
                                    if dx != 0 or dy != 0:
                                        # Create outline by pasting emoji with offset
                                        outline_img = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
                                        outline_img.paste(content, (actual_x + dx, actual_y + dy))
                                        pil_image = Image.alpha_composite(pil_image.convert('RGBA'), outline_img).convert('RGB')
                        
                        # Paste emoji
                        if content.mode == 'RGBA':
                            pil_image.paste(content, (actual_x, actual_y), content)
                        else:
                            pil_image.paste(content, (actual_x, actual_y))
                    
                    elif item_type == 'text':
                        # Draw text outline if enabled
                        if self.config.outline_enabled:
                            for dx in range(-self.config.outline_width, self.config.outline_width + 1):
                                for dy in range(-self.config.outline_width, self.config.outline_width + 1):
                                    if dx != 0 or dy != 0:
                                        draw.text((actual_x + dx, actual_y + dy), content, 
                                                font=self.font, fill=outline_color)
                        
                        # Draw main text
                        draw.text((actual_x, actual_y), content, font=self.font, fill=text_color)
        
        else:
            # Fallback to regular text rendering without emoji
            current_y = y
            for line in text_lines:
                # Calculate line position
                line_width, line_height = self._get_text_size(line, self.font)
                
                # Adjust x position for center/right alignment
                if self.config.position.endswith('center'):
                    line_x = (frame_width - line_width) // 2
                elif self.config.position.endswith('right'):
                    line_x = frame_width - line_width - self.config.margin
                else:
                    line_x = x
                
                # Draw outline if enabled
                if self.config.outline_enabled:
                    for dx in range(-self.config.outline_width, self.config.outline_width + 1):
                        for dy in range(-self.config.outline_width, self.config.outline_width + 1):
                            if dx != 0 or dy != 0:
                                draw.text((line_x + dx, current_y + dy), line, 
                                        font=self.font, fill=outline_color)
                
                # Draw main text
                draw.text((line_x, current_y), line, font=self.font, fill=text_color)
                
                # Move to next line
                current_y += int(line_height * self.config.line_spacing)
        
        # Convert back to BGR
        frame_with_text = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return frame_with_text
    
    def update_config(self, config: TextOverlayConfig):
        """Update configuration and reload font if needed"""
        if (config.font_family != self.config.font_family or 
            config.font_size != self.config.font_size):
            self.config = config
            self.font = self._load_font()
        else:
            self.config = config