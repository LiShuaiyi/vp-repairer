import os
from PIL import Image
import cairosvg

# violation or repair
tag = "violated" # or "repaired"
# tag = "repaired" # or "violated"
# Directory containing the SVG files
svg_dir = '.'
png_dir = './png_frames'
gif_output_path = f'./rin1_{tag}.gif'

# Ensure the output directory for PNG frames exists
os.makedirs(png_dir, exist_ok=True)

# Convert SVGs ending with "_violation.svg" to PNGs
frame_files = []
for file_name in sorted(os.listdir(svg_dir)):
    if file_name.endswith(f'_{tag}.svg'):
        svg_path = os.path.join(svg_dir, file_name)
        png_path = os.path.join(png_dir, f"{os.path.splitext(file_name)[0]}.png")
        frame_files.append(png_path)

        # Convert SVG to PNG
        cairosvg.svg2png(url=svg_path, write_to=png_path)

# Create GIF from PNG frames
if frame_files:
    frames = [Image.open(frame) for frame in frame_files]
    frames[0].save(
        gif_output_path,
        save_all=True,
        append_images=frames[1:],
        duration=200,  # 100 ms per frame
        loop=0  # Infinite loop
    )
    print(f"GIF created at {gif_output_path}")
else:
    print(f"No `_{tag}.svg` files found to process.")

# Optional: Clean up PNG frames after GIF creation (uncomment if needed)
for png_file in frame_files:
    os.remove(png_file)
# os.rmdir(png_dir)  # Remove the directory if empty
