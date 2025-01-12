import streamlit as st
import yt_dlp
import pandas as pd
import os
import requests
import time
from moviepy.editor import VideoFileClip
import subprocess
import shutil

# Streamlit app title
st.title("YouTube Shorts and TikTok Video Downloader")

# Function to download YouTube Shorts
def yt_shorts_downloader(url, save_path):
    ydl_opts = {
        'format': 'best',
        'outtmpl': os.path.join(save_path, '%(id)s.%(ext)s'),
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

# Function to check Apify run status
def check_apify_run_status(run_id, api_token):
    url = f"https://api.apify.com/v2/actor-runs/{run_id}?token={api_token}"
    while True:
        response = requests.get(url)
        if response.status_code == 200:
            run_data = response.json()
            status = run_data.get('data', {}).get('status')
            if status == 'SUCCEEDED':
                st.success("Apify run succeeded!")
                break
            elif status == 'FAILED':
                st.error("Apify run failed!")
                break
            elif status == 'RUNNING':
                st.info("Run is still in progress...")
            else:
                st.warning(f"Unknown status: {status}")
            time.sleep(5)
        else:
            st.error(f"Error fetching run status: {response.status_code}, {response.text}")
            break

# Function to validate URLs from Excel
def validate_urls_from_excel(file_name: str, sheet_name: str, column_name: str):
    try:
        df = pd.read_excel(file_name, sheet_name=sheet_name)
        urls = df[column_name].tolist()
        for url in urls:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code != 200:
                    st.warning(f"Invalid URL (status code {response.status_code}): {url}")
            except requests.exceptions.RequestException as e:
                st.warning(f"Error accessing URL: {url}, Error: {e}")
    except Exception as e:
        st.error(f"Error reading the Excel file or processing URLs: {e}")

# Function to convert media files to GIFs
def convert_to_gif(media_file, max_duration=10, fps=10, output_dir='/content/gifs'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    try:
        with VideoFileClip(media_file) as clip:
            if clip.duration > max_duration:
                clip = clip.subclip(0, max_duration)
            clip = clip.set_fps(fps)
            output_gif_path = os.path.join(output_dir, os.path.splitext(os.path.basename(media_file))[0] + '.gif')
            clip.write_gif(output_gif_path)
            st.success(f"Converted {media_file} to {output_gif_path}")
            return output_gif_path
    except Exception as e:
        st.error(f"Failed to convert {media_file}: {e}")
        return None

# Function to download and trim video
def download_and_trim_video(url, output_dir='/content/videos', duration=10):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    video_filename = os.path.basename(url)
    output_path = os.path.join(output_dir, video_filename)
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'outtmpl': output_path,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        st.error(f"Error downloading video: {e}")
        return None

    trimmed_output = os.path.join(output_dir, f"trimmed_{video_filename}")
    command = [
        'ffmpeg',
        '-i', output_path,
        '-t', str(duration),
        '-c', 'copy',
        '-y',
        trimmed_output
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            st.error(f"Error running ffmpeg: {result.stderr}")
        else:
            st.success(f"Video trimmed and saved as {trimmed_output}")
        os.remove(output_path)
        os.rename(trimmed_output, output_path)
        return output_path
    except Exception as e:
        st.error(f"Failed to trim the video: {e}")
        return None

# Function to resize GIFs
def resize_gif(input_gif, max_size_mb=1.99, processed_dir='/content/processed_gifs'):
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    clip = VideoFileClip(input_gif)
    fps = clip.fps
    current_size = os.path.getsize(input_gif)
    if current_size > max_size_mb * 1024 * 1024:
        reduction_factor = 0.95
        while current_size > max_size_mb * 1024 * 1024:
            new_duration = clip.duration * reduction_factor
            clip = clip.subclip(0, new_duration)
            clip.write_gif("temp_resized.gif", fps=fps)
            current_size = os.path.getsize("temp_resized.gif")
            os.remove("temp_resized.gif")
            reduction_factor *= 0.95
        base, ext = os.path.splitext(os.path.basename(input_gif))
        output_gif = os.path.join(processed_dir, f"{base}{ext}")
        clip.write_gif(output_gif, fps=fps)
        st.success(f"Resized GIF saved as {output_gif}, size: {current_size/1024/1024:.2f} MB")
    else:
        shutil.copy(input_gif, processed_dir)

# Main logic to process videos from Excel
def process_videos_from_excel(input_excel, sheet_name, output_dir='/content/gifs', youtube_shorts_folder='/content/youtube_shorts'):
    df = pd.read_excel(input_excel, sheet_name=sheet_name)
    for index, row in df.iterrows():
        video_url = row["Gcs Url"]
        st.write(f"Processing TikTok video URL: {video_url}")
        video_path = download_and_trim_video(video_url)
        if video_path:
            convert_to_gif(video_path, output_dir=output_dir)

    if os.path.exists(youtube_shorts_folder):
        for video_file in os.listdir(youtube_shorts_folder):
            video_path = os.path.join(youtube_shorts_folder, video_file)
            st.write(f"Processing YouTube Shorts video: {video_path}")
            if os.path.isfile(video_path):
                convert_to_gif(video_path, output_dir=output_dir)

# Streamlit user input for file upload
uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.write("Data from Excel:", df)

    # Input parameters for TikTok and YouTube Shorts
    if st.button("Start Processing"):
        input_excel = uploaded_file
        sheet_name = 'Sheet1'  # Replace with the sheet name
        process_videos_from_excel(input_excel, sheet_name)
        st.success("Processing completed!")

# Clean up files button
if st.button("Clean Up Files"):
    # Clean the files from the content directory
    shutil.rmtree('/content/*', ignore_errors=True)
    st.success("Cleaned up files from the content directory.")