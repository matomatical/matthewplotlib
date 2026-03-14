import numpy as np
import subprocess
import matthewplotlib as mp

# Settings
WIDTH, HEIGHT = 128, 96
FRAME_SIZE = WIDTH * HEIGHT * 3  # 3 bytes per pixel (RGB)

# FFmpeg command extracted from source (Mac-specific 'avfoundation')
cmd = [
    'ffmpeg',
    '-hide_banner',
    '-loglevel', 'error',
    '-f', 'avfoundation',       # Input format (macOS)
    '-framerate', '30',         # Input fps
    '-video_size', '1280x720',  # Input resolution
    '-pixel_format', 'uyvy422', # pixel format again?
    '-i', '0',                  # Camera index
    '-vf', f'scale={WIDTH}:{HEIGHT}', # Scale to target resolution
    '-r', '30',                 # Output fps (User requested 30)
    '-f', 'rawvideo',
    '-pix_fmt', 'rgb24',        # Pixel format
    '-'                         # Output to stdout
]

# Start FFmpeg process
process = subprocess.Popen(cmd, stdout=subprocess.PIPE)

# Read frames loop
first = True
while True:
    # Read exactly one frame's worth of bytes
    frame_data = process.stdout.read(FRAME_SIZE)
    
    if len(frame_data) != FRAME_SIZE:
        break

    image = np.frombuffer(
        frame_data,
        dtype=np.uint8,
    ).reshape(
        (HEIGHT, WIDTH, 3),
    )
        
    plot = mp.image(image)
    if first:
        first = False
        print(plot)
    else:
        print(f"{-plot}{plot}")
