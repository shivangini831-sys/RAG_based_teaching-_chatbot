import os
import subprocess

os.makedirs("audios", exist_ok=True)

files = os.listdir("videos")

for file in files:
    if not file.lower().endswith(".mp4"):
        continue

    print("Processing:", file)

    # Extract tutorial number safely
    if "#" in file:
        tutorial_number = file.split("#")[-1].replace(".mp4", "").strip()
    else:
        tutorial_number = "unknown"

    # Remove extension for clean name
    file_name = os.path.splitext(file)[0]

    output_path = f"audios/{tutorial_number}_{file_name}.mp3"

    subprocess.run([
        "ffmpeg",
        "-y",                     # auto-overwrite
        "-i", f"videos/{file}",
        output_path
    ])
