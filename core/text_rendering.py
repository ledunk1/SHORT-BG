from PIL import Image, ImageDraw, ImageFont
import os
import re

EMOJI_FOLDER = "emoji/"

def load_emoji(emoji_char, emoji_size):
    """Memuat gambar emoji dari file."""
    emoji_chars = list(emoji_char)
    emoji_imgs = []
    for char in emoji_chars:
        unicode_hex = f"{ord(char):04X}"
        emoji_files = [
            os.path.join(EMOJI_FOLDER, f"{unicode_hex}.png"),
            os.path.join(EMOJI_FOLDER, f"{unicode_hex}FE0F.png")
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

def get_text_dimensions(draw, text, font):
    """Mendapatkan dimensi text."""
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]

def get_emoji_pattern():
    """Mendapatkan pattern regex untuk emoji."""
    return re.compile(
        "["
        "\U0001F600-\U0001F64F" "\U0001F300-\U0001F5FF" "\U0001F680-\U0001F6FF"
        "\U0001F700-\U0001F77F" "\U0001F780-\U0001F7FF" "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF" "\U0001FA00-\U0001FA6F" "\U0001FA70-\U0001FAFF"
        "\U00002600-\U000027BF" "\U0001F1E6-\U0001F1FF" "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251" "]+", flags=re.UNICODE,
    )

def calculate_content_width(parts, draw, font, emoji_size):
    """Menghitung lebar total konten (text + emoji)."""
    emoji_pattern = get_emoji_pattern()
    total_width = 0
    
    for part in parts:
        if emoji_pattern.fullmatch(part):
            total_width += emoji_size
        else:
            text_width, _ = get_text_dimensions(draw, part, font)
            total_width += text_width
    
    return total_width

def smart_text_wrap(text, draw, font, max_width, emoji_size=80):
    """
    Membungkus text dengan emoji secara cerdas berdasarkan lebar maksimum.
    Mendukung auto-detect frame dan paragraf otomatis.
    """
    emoji_pattern = get_emoji_pattern()
    
    # Split text menjadi kata-kata, mempertahankan emoji
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
    
    # Bungkus kata-kata menjadi baris
    lines = []
    current_line = []
    current_line_width = 0
    
    for word in words:
        # Hitung lebar word (text atau emoji)
        if emoji_pattern.fullmatch(word):
            word_width = emoji_size
        else:
            word_width, _ = get_text_dimensions(draw, word, font)
        
        # Tambahkan spasi jika bukan kata pertama di baris
        space_width = 0
        if current_line and not emoji_pattern.fullmatch(word):
            space_width, _ = get_text_dimensions(draw, " ", font)
        
        # Cek apakah word muat di baris saat ini
        if current_line_width + space_width + word_width <= max_width:
            if current_line and not emoji_pattern.fullmatch(word):
                current_line.append(" ")
                current_line_width += space_width
            current_line.append(word)
            current_line_width += word_width
        else:
            # Word tidak muat, buat baris baru
            if current_line:
                lines.append("".join(current_line))
            current_line = [word]
            current_line_width = word_width
    
    # Tambahkan baris terakhir
    if current_line:
        lines.append("".join(current_line))
    
    return lines

def render_text_with_emoji_multiline(draw, lines, font, canvas_width, canvas_height, 
                                   start_y, emoji_size=80, line_spacing=10):
    """
    Merender multiple lines text dengan emoji.
    Auto-detect jika text melebihi frame dan sesuaikan.
    """
    emoji_pattern = get_emoji_pattern()
    rendered_lines = []
    current_y = start_y
    
    # Hitung tinggi total yang dibutuhkan
    line_height = font.size + line_spacing
    total_height = len(lines) * line_height
    
    # Auto-adjust jika melebihi frame
    if current_y + total_height > canvas_height:
        # Sesuaikan posisi Y agar text muat
        current_y = max(10, canvas_height - total_height - 20)
    
    for line in lines:
        if current_y + line_height > canvas_height:
            break  # Stop jika sudah melebihi frame
            
        # Split line menjadi parts (text dan emoji)
        parts = re.split(f"({emoji_pattern.pattern})", line)
        
        # Hitung lebar total line
        total_width = calculate_content_width(parts, draw, font, emoji_size)
        
        # Auto-adjust jika melebihi lebar frame
        if total_width > canvas_width - 40:  # 40px margin
            # Scale down emoji jika perlu
            adjusted_emoji_size = min(emoji_size, int(emoji_size * (canvas_width - 40) / total_width))
            total_width = calculate_content_width(parts, draw, font, adjusted_emoji_size)
        else:
            adjusted_emoji_size = emoji_size
        
        # Posisi X untuk center alignment
        x_start = (canvas_width - total_width) // 2
        
        # Render parts
        rendered_items = []
        x_offset = 0
        
        for part in parts:
            if emoji_pattern.fullmatch(part):
                emoji_img = load_emoji(part, adjusted_emoji_size)
                if emoji_img:
                    rendered_items.append(('emoji', emoji_img, x_offset))
                    x_offset += adjusted_emoji_size
            else:
                if part.strip():  # Skip empty strings
                    rendered_items.append(('text', part, x_offset))
                    text_width, _ = get_text_dimensions(draw, part, font)
                    x_offset += text_width
        
        rendered_lines.append({
            'items': rendered_items,
            'x_start': x_start,
            'y': current_y,
            'emoji_size': adjusted_emoji_size
        })
        
        current_y += line_height
    
    return rendered_lines

def render_text_with_emoji(draw, text, font, canvas_width, emoji_size=80):
    """
    Merender teks dan emoji dalam satu baris (backward compatibility).
    """
    emoji_pattern = get_emoji_pattern()
    parts = re.split(f"({emoji_pattern.pattern})", text)
    rendered_items, x_offset, total_width = [], 0, 0
    
    for part in parts:
        if emoji_pattern.fullmatch(part): 
            total_width += emoji_size
        else: 
            text_width, _ = get_text_dimensions(draw, part, font)
            total_width += text_width
    
    x_start = (canvas_width - total_width) // 2
    
    for part in parts:
        if emoji_pattern.fullmatch(part):
            emoji_img = load_emoji(part, emoji_size)
            if emoji_img:
                rendered_items.append((emoji_img, x_offset))
                x_offset += emoji_size
        else:
            text_width, _ = get_text_dimensions(draw, part, font)
            rendered_items.append((part, x_offset))
            x_offset += text_width
    
    return rendered_items, x_start

def wrap_text(text, font, max_width, draw):
    """Memecah teks panjang menjadi beberapa baris (legacy function)."""
    return smart_text_wrap(text, draw, font, max_width)