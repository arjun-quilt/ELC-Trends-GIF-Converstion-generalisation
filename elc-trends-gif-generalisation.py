import yt_dlp
import pandas as pd
import os
import requests
import time
import shutil
from moviepy.editor import VideoFileClip
import pandas as pd
import subprocess
from yt_dlp.utils import DownloadError
from google.cloud import storage
import streamlit as st  # Import Streamlit
import tempfile



# Extract the secret
gcp_secret = st.secrets["gcp_secret"]

# Write the secret to a temporary file
with tempfile.NamedTemporaryFile(delete=False, mode="w") as temp_file:
    temp_file.write(gcp_secret)
    temp_file_path = temp_file.name

# Set the environment variable to the temporary file path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_file_path
# Now proceed with your GCS operations

storage_client = storage.Client()
bucket = storage_client.bucket('tiktok-actor-content')

def yt_shorts_downloader(urls, bucket_name):
    # Ensure URLs is a list
    if not isinstance(urls, list):
        raise ValueError("The URLs parameter should be a list of strings.")

    # Initialize Google Cloud Storage client
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Iterate through the list of URLs
    for url in urls:
        # Set options for yt-dlp
        ydl_opts = {
            'format': 'best',
            'outtmpl': '-',  # Output to stdout (this avoids saving it locally)
            'quiet': True,    # Suppress yt-dlp's output
        }

        # Download video into memory and upload to GCS
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(url, download=False)
            video_id = result['id']
            video_ext = result['ext']

            # Stream the video to memory
            video_data = io.BytesIO(ydl.urlopen(url).read())

            # Define the destination file name in the bucket
            destination_blob = f"{video_id}.{video_ext}"

            # Upload to GCS
            upload_to_gcs(bucket, video_data, destination_blob)


def upload_to_gcs(bucket, video_data, destination_blob):
    """
    Uploads the video stream directly to Google Cloud Storage.

    :param bucket: GCS bucket object
    :param video_data: BytesIO stream containing the video data
    :param destination_blob: The destination file name in the GCS bucket
    """
    blob = bucket.blob(destination_blob)
    blob.upload_from_file(video_data)
    print(f"Video uploaded to {destination_blob} in bucket {bucket.name}.")



# Streamlit app title
st.title("TikTok and YouTube Shorts Downloader")

# File uploader for Excel file
uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

# Initialize tiktok_videos and youtube_shorts to avoid NameError
tiktok_videos = []
youtube_shorts = []

if uploaded_file is not None:
    # Read the Excel file
    df = pd.read_excel(uploaded_file, sheet_name='Master_Sheet')  # Use uploaded file
    urls = df['Links'].tolist()

    # TikTok Videos: Last part of the URL contains only numbers
    tiktok_videos = [url for url in urls if url.split("/")[-1].isdigit()]

    # YouTube Shorts: Last part of the URL contains both numbers and letters
    youtube_shorts = [url for url in urls if any(c.isalpha() for c in url.split("/")[-1]) and any(c.isdigit() for c in url.split("/")[-1])]

    # Button to start downloading
    if st.button("Download Videos"):
        yt_shorts_downloader(
            urls=youtube_shorts,
            bucket_name="tiktok-actor-content"
        )
        st.success("Videos downloaded successfully!")

# Input parameters
input_params = {
    "disableCheerioBoost": False,
    "disableEnrichAuthorStats": False,
    "resultsPerPage": 1,
    "searchSection": "/video",
    "shouldDownloadCovers": True,
    "shouldDownloadSlideshowImages": False,
    "shouldDownloadVideos": True,
    "maxProfilesPerQuery": 10,
    "tiktokMemoryMb": "default",
    "urls": tiktok_videos  # Only use tiktok videos to process in apify
}

# API token
API_TOKEN = "apify_api_VUQNA5xFO4IwieTeWX7HmKUYnNZOnw0c2tgk"

# Task ID for TikTok scraper
TASK_ID = "H70fR5ndjUD0loq5H"

# API request to run the task
# task_url = f"https://api.apify.com/v2/actor-tasks/{TASK_ID}/runs?token={API_TOKEN}"
task_url = f"https://api.apify.com/v2/actor-tasks/quilt-org~tiktok-orchestrator-elc-trends-gifs/runs?token={API_TOKEN}"
headers = {"Content-Type": "application/json"}

response = requests.post(task_url, json={"input": input_params}, headers=headers)

# Print the response
st.write(f"Response Status Code: {response.status_code}")
st.write(response.json())

data = response.json()
run_id = data['data']['id']
print(run_id)

def check_apify_run_status(run_id, api_token):
    url = f"https://api.apify.com/v2/actor-runs/{run_id}?token={api_token}"

    while True:
        response = requests.get(url)

        if response.status_code == 200:
            run_data = response.json()

            # Extract the status field
            status = run_data.get('data', {}).get('status')

            if status == 'SUCCEEDED':
                print("SUCCEEDED")
                break
            elif status == 'FAILED':
                print("FAILED")
                break
            elif status == 'RUNNING':
                print("Run is still in progress...")
            else:
                print(f"Unknown status: {status}")

            # Wait for a few seconds before checking again
            time.sleep(5)
        else:
            print(f"Error fetching run status: {response.status_code}, {response.text}")
            break

# Replace with your run ID and Apify API token
data = response.json()
run_id = data['data']['id']
print(run_id)
api_token = 'apify_api_VUQNA5xFO4IwieTeWX7HmKUYnNZOnw0c2tgk'  # Replace with your API token

check_apify_run_status(run_id, api_token)

input_list_of_dicts = df.to_dict(orient="records")

# Loop through each row (input_dict) and add the GCS URL
for input_dict in input_list_of_dicts:
    clean_url = input_dict["Links"]
    video_id = clean_url.split("?")[0].split("/")[-1]
    input_dict["Gcs Url"] = f"https://storage.googleapis.com/tiktok-actor-content/{video_id}.mp4"
    input_dict["Gif Url"] = f"https://storage.googleapis.com/tiktok-actor-content/gifs_20240419/{video_id}.gif"

# Optionally, you can save this updated data to a new Excel file
output_df = pd.DataFrame(input_list_of_dicts)
output_file_name = 'updated_tiktok_urls.xlsx'  # The name of the updated file
output_df.to_excel(output_file_name, index=False)

st.write(f"Updated Excel file saved as {output_file_name}")

import pandas as pd
import requests

def validate_urls_from_excel(file_name: str, sheet_name: str, column_name: str):
    try:
        # Read the Excel file
        df = pd.read_excel(file_name, sheet_name=sheet_name)

        # Get the list of URLs from the specified column
        urls = df[column_name].tolist()

        # Validate each URL
        for url in urls:
            try:
                response = requests.get(url, timeout=5)  # Set a timeout to handle slow responses
                if response.status_code != 200:
                    print(f"Invalid URL (status code {response.status_code}): {url}")
            except requests.exceptions.RequestException as e:
                print(f"Error accessing URL: {url}, Error: {e}")

    except Exception as e:
        print(f"Error reading the Excel file or processing URLs: {e}")


# Example usage
file_name = '/content/updated_tiktok_urls.xlsx'  # Replace with your file path
sheet_name = 'Sheet1'  # Replace with your sheet name
column_name = 'Gcs Url'  # Replace with your column name containing URLs

validate_urls_from_excel(file_name, sheet_name, column_name)

# Function to convert media files to GIFs
def convert_to_gif(media_file, max_duration=10, fps=10, output_dir='/content/gifs'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        with VideoFileClip(media_file) as clip:
            # If the video is longer than `max_duration`, use only the first `max_duration` seconds
            if clip.duration > max_duration:
                clip = clip.subclip(0, max_duration)

            clip = clip.set_fps(fps)

            output_gif_path = os.path.join(output_dir, os.path.splitext(os.path.basename(media_file))[0] + '.gif')
            clip.write_gif(output_gif_path)
            print(f"Converted {media_file} to {output_gif_path}")
            return output_gif_path  # Return the GIF path
    except Exception as e:
        print(f"Failed to convert {media_file}: {e}")
        return None

# Function to download video from GCS URL

def download_and_trim_video(url, output_dir='/content/videos', duration=10):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Extract the filename from the URL and set the output file name
    video_filename = os.path.basename(url)
    output_path = os.path.join(output_dir, video_filename)

    # Step 1: Download the video using yt-dlp
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',  # Adjust format if needed
        'outtmpl': output_path,  # Save with the same name
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # Step 2: Use ffmpeg to trim the first 10 seconds and save to a new temporary file
    trimmed_output = os.path.join(output_dir, f"trimmed_{video_filename}")
    command = [
        'ffmpeg',
        '-i', output_path,  # Input file
        '-t', str(duration),  # Duration in seconds
        '-c', 'copy',  # Copy the original codec (avoids re-encoding)
        '-y',  # Force overwriting the temporary file
        trimmed_output  # Output file (new file for the trimmed video)
    ]

    try:
        # Run the ffmpeg command to trim the video
        result = subprocess.run(command, capture_output=True, text=True)

        # Check if ffmpeg encountered any errors
        if result.returncode != 0:
            print(f"Error running ffmpeg: {result.stderr}")
        else:
            print(f"Video trimmed and saved as {trimmed_output}")

        # Remove the untrimmed video after trimming
        os.remove(output_path)

        # Rename the trimmed video to the original file name (replace original file)
        os.rename(trimmed_output, output_path)

        return output_path  # Return the path to the trimmed video

    except Exception as e:
        print(f"Failed to trim the video: {e}")
        return None


# Function to resize GIFs
def resize_gif(input_gif, max_size_mb=1.99, processed_dir='/content/processed_gifs'):
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    clip = VideoFileClip(input_gif)
    fps = clip.fps  # Frames per second
    current_size = os.path.getsize(input_gif)

    if current_size > max_size_mb * 1024 * 1024:
        reduction_factor = 0.95  # Initial reduction by 5%
        while current_size > max_size_mb * 1024 * 1024:
            new_duration = clip.duration * reduction_factor
            clip = clip.subclip(0, new_duration)
            clip.write_gif("temp_resized.gif", fps=fps)
            current_size = os.path.getsize("temp_resized.gif")
            os.remove("temp_resized.gif")  # Clearing the temporary file
            reduction_factor *= 0.95  # Reduce the duration further

        # Save the resized GIF in the processed directory
        base, ext = os.path.splitext(os.path.basename(input_gif))
        output_gif = os.path.join(processed_dir, f"{base}{ext}")
        clip.write_gif(output_gif, fps=fps)
        print(f"Resized GIF saved as {output_gif}, size: {current_size/1024/1024:.2f} MB")
    else:
        # If not resizing, just copy the original GIF to the processed directory
        shutil.copy(input_gif, processed_dir)

# Main logic to process an Excel file and convert videos to GIFs
import os

def process_videos_from_excel(input_excel, sheet_name, output_dir='/content/gifs'):
    df = pd.read_excel(input_excel, sheet_name=sheet_name)

    # Assume GCS URLs are in the "Gcs Url" column
    for index, row in df.iterrows():
        video_url = row["Gcs Url"]  # Replace with the correct column name in your Excel
        print(f"Processing video URL: {video_url}")

        # Download the video from tiktok
        video_path = download_and_trim_video(video_url)

        if video_path:
            # Convert to GIF
            gif_path = convert_to_gif(video_path, output_dir=output_dir)

            # Process the converted GIF to make it smaller than 2 MB
            if gif_path:
                resize_gif(gif_path)

                # Delete the GIF after resizing
                if os.path.exists(gif_path):
                    os.remove(gif_path)
                    print(f"Deleted GIF: {gif_path}")

            # Delete the video after converting to GIF
            if os.path.exists(video_path):
                os.remove(video_path)
                print(f"Deleted video: {video_path}")


# Example usage
input_excel = "/content/updated_tiktok_urls.xlsx"   # Replace with the actual path to your Excel file
sheet_name = 'Sheet1'  # Replace with the sheet name
process_videos_from_excel(input_excel, sheet_name)

# Example usage
if st.button("Validate URLs"):
    validate_urls_from_excel(output_file_name, 'Sheet1', 'Gcs Url')
    st.success("URL validation completed!")

# Main logic to process an Excel file and convert videos to GIFs
if st.button("Process Videos"):
    process_videos_from_excel(output_file_name, 'Sheet1')
    st.success("Video processing completed!")