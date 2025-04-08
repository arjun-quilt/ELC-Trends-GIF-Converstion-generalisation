# app.py
import streamlit as st
import pandas as pd
import os
import urllib.request
from moviepy.editor import VideoFileClip
import httpx
import asyncio
from google.cloud import storage
import json
import tempfile
import shutil
import requests
import time
from datetime import datetime

# Initialize session state
if 'apify_status' not in st.session_state:
    st.session_state.apify_status = None
if 'run_id' not in st.session_state:
    st.session_state.run_id = None
if 'dataset_id' not in st.session_state:
    st.session_state.dataset_id = None
if 'input_list_of_dicts' not in st.session_state:
    st.session_state.input_list_of_dicts = None
if 'country_name' not in st.session_state:
    st.session_state.country_name = None
if 'final_results' not in st.session_state:
    st.session_state.final_results = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'start_time' not in st.session_state:
    st.session_state.start_time = None
if 'end_time' not in st.session_state:
    st.session_state.end_time = None
if 'apify_start_time' not in st.session_state:
    st.session_state.apify_start_time = None
if 'apify_end_time' not in st.session_state:
    st.session_state.apify_end_time = None
if 'gif_start_time' not in st.session_state:
    st.session_state.gif_start_time = None
if 'gif_end_time' not in st.session_state:
    st.session_state.gif_end_time = None

# Extract the secret and create temporary credentials file
gcp_secret = st.secrets["gcp_secret"]
with tempfile.NamedTemporaryFile(delete=False, mode="w") as temp_file:
    temp_file.write(gcp_secret)
    temp_file_path = temp_file.name

# Set the environment variable to the temporary file path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_file_path

# Function to run the Apify actor task
async def run_actor_task(data: dict) -> dict:
    url = "https://api.apify.com/v2/actor-tasks/eKYRHMIgvYqAlh1r3/runs?token=apify_api_VUQNA5xFO4IwieTeWX7HmKUYnNZOnw0c2tgk"
    headers = {"Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, json=data, headers=headers)
        response.raise_for_status()
        return response

# Function to get items from the dataset
async def get_items(dataset_id: str) -> dict:
    url = f"https://api.apify.com/v2/datasets/{dataset_id}/items?clean=true"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.json()

# Function to convert video to GIF and return the path
def convert_to_gif(media_file, video_id, max_duration=2, fps=3):
    temp_gif_path = f"{video_id}.gif"
    max_size_mb = 1.99  # Maximum size in MB
    
    with VideoFileClip(media_file) as clip:
        if clip.duration > max_duration:
            clip = clip.subclip(0, max_duration)
        
        # Initial conversion
        clip = clip.set_fps(fps)
        clip.write_gif(temp_gif_path)
        
        # Check size and resize if needed
        current_size = os.path.getsize(temp_gif_path)
        if current_size > max_size_mb * 1024 * 1024:
            reduction_factor = 0.95  # Initial reduction by 5%
            while current_size > max_size_mb * 1024 * 1024 and clip.duration > 1:
                new_duration = clip.duration * reduction_factor
                clip = clip.subclip(0, new_duration)
                clip.write_gif(temp_gif_path, fps=fps)
                current_size = os.path.getsize(temp_gif_path)
                reduction_factor *= 0.95  # Reduce the duration further
    
    return temp_gif_path

# Function to upload GIF to GCS
def upload_gif_to_gcs(local_file_path: str, video_id: str, bucket_name: str, gcs_folder: str):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    gcs_blob_path = f"{gcs_folder}/{video_id}.gif"
    blob = bucket.blob(gcs_blob_path)
    blob.upload_from_filename(local_file_path)
    return f"https://storage.googleapis.com/{bucket_name}/{gcs_blob_path}"

def check_run_status(run_id: str, api_token: str, status_placeholder) -> bool:
    """Check status of a single Apify run (synchronous version)"""
    url = f"https://api.apify.com/v2/actor-runs/{run_id}?token={api_token}"
    retry_delay = 5

    while True:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                run_data = response.json()
                status = run_data.get('data', {}).get('status')
                
                if status == 'SUCCEEDED':
                    st.session_state.apify_end_time = datetime.now()
                    apify_duration = st.session_state.apify_end_time - st.session_state.apify_start_time
                    hours, remainder = divmod(apify_duration.total_seconds(), 3600)
                    minutes, seconds = divmod(remainder, 60)
                    status_placeholder.success(f"Run {run_id}: SUCCEEDED (Time taken: {int(hours)}h {int(minutes)}m {int(seconds)}s)")
                    st.session_state.apify_status = 'SUCCEEDED'
                    return True
                elif status == 'FAILED':
                    st.session_state.apify_end_time = datetime.now()
                    status_placeholder.error(f"Run {run_id}: FAILED")
                    st.session_state.apify_status = 'FAILED'
                    return False
                elif status == 'RUNNING':
                    current_time = datetime.now()
                    running_duration = current_time - st.session_state.apify_start_time
                    hours, remainder = divmod(running_duration.total_seconds(), 3600)
                    minutes, seconds = divmod(remainder, 60)
                    status_placeholder.info(f"Run {run_id}: Still running... (Running for: {int(hours)}h {int(minutes)}m {int(seconds)}s)")
                    st.session_state.apify_status = 'RUNNING'
                    time.sleep(5)
                else:
                    status_placeholder.warning(f"Run {run_id}: Unknown status - {status}")
                    st.session_state.apify_status = 'UNKNOWN'
                    return False
            else:
                raise Exception(f"HTTP {response.status_code}")

        except Exception as e:
            status_placeholder.warning(f"Run {run_id}: Request failed - {str(e)}")
            time.sleep(retry_delay)

# Streamlit UI
st.title("TikTok Video to GIF Converter")

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
country_name = st.text_input("Enter the country name (sheet name)")

if st.button("Process"):
    if uploaded_file and country_name:
        # Store data in session state
        st.session_state.country_name = country_name
        st.session_state.input_list_of_dicts = pd.read_excel(uploaded_file, sheet_name=country_name).to_dict(orient="records")

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
            "postURLs": [row["Links"] for row in st.session_state.input_list_of_dicts],
        }

        st.write("Running the Apify actor task...")
        response = asyncio.run(run_actor_task(input_params))
        st.session_state.run_id = response.json()["data"]["id"]
        st.session_state.dataset_id = response.json()["data"]["defaultDatasetId"]
        
        # Create a placeholder for status updates
        status_placeholder = st.empty()
        
        # Check run status
        api_token = "apify_api_VUQNA5xFO4IwieTeWX7HmKUYnNZOnw0c2tgk"
        check_run_status(st.session_state.run_id, api_token, status_placeholder)

# If we have a running Apify task, show its status
if st.session_state.apify_status == 'RUNNING':
    status_placeholder = st.empty()
    check_run_status(st.session_state.run_id, "apify_api_VUQNA5xFO4IwieTeWX7HmKUYnNZOnw0c2tgk", status_placeholder)

# If Apify task succeeded, proceed with GIF conversion
if st.session_state.apify_status == 'SUCCEEDED' and not st.session_state.processing_complete:
    try:
        st.write("Apify task completed successfully. Starting GIF conversion...")
        
        # Store GIF conversion start time
        st.session_state.gif_start_time = datetime.now()
        st.write(f"GIF conversion started at: {st.session_state.gif_start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        st.write("Fetching items from the dataset...")
        dataset = asyncio.run(get_items(st.session_state.dataset_id))
        st.write(f"Fetched {len(dataset)} items from the dataset.")

        all_items_dict = {}
        for raw_row in dataset:
            try:
                original_url = raw_row["submittedVideoUrl"]
                all_items_dict[original_url] = {
                    "Gcs Url": raw_row["gcsMediaUrls"][0]
                }
            except Exception as e:
                st.error(f"Error processing row: {raw_row} - {e}")

        for input_dict in st.session_state.input_list_of_dicts:
            clean_url = input_dict["Links"]
            video_id = clean_url.split("?")[0].split("/")[-1]
            input_dict["Gcs Url"] = f"https://storage.googleapis.com/tiktok-actor-content/{video_id}.mp4"

        output_df = pd.DataFrame(st.session_state.input_list_of_dicts)
        output_df.to_csv(f"{st.session_state.country_name}_duration.csv", index=False, encoding="utf_8_sig")
        st.write(f"Generated CSV file: {st.session_state.country_name}_duration.csv")

        # Download videos and convert to GIFs
        df = pd.read_csv(f"{st.session_state.country_name}_duration.csv")
        list_of_dicts = df.to_dict(orient="records")

        # Initialize overall progress bar
        total_videos = len(list_of_dicts)
        overall_progress = st.progress(0, text=f"Processing {total_videos} videos...")

        # Process each video
        for i, row in enumerate(list_of_dicts):
            try:
                # Update progress
                progress_percent = (i + 1) / total_videos
                overall_progress.progress(progress_percent, text=f"Processing video {i + 1}/{total_videos}...")

                # Download video
                video_url = row["Gcs Url"]
                video_path = f"temp_video_{i}.mp4"
                urllib.request.urlretrieve(video_url, video_path)

                # Convert to GIF
                clip = VideoFileClip(video_path)
                gif_path = f"output_{i}.gif"
                clip.write_gif(gif_path, fps=10)

                # Upload to GCS
                bucket_name = "tiktok-actor-content"
                gcs_path = f"{st.session_state.country_name}_gifs/{os.path.basename(gif_path)}"
                upload_gif_to_gcs(gif_path, video_id, bucket_name, gcs_path)

                # Update row with GIF URL
                row["Gif Url"] = f"https://storage.googleapis.com/{bucket_name}/{gcs_path}"

                # Clean up temporary files
                os.remove(video_path)
                os.remove(gif_path)

            except Exception as e:
                st.error(f"Error processing video {i + 1}: {str(e)}")
                continue

        # Store final results in session state
        st.session_state.final_results = pd.DataFrame(list_of_dicts)
        st.session_state.final_results.to_csv(f"{st.session_state.country_name}_trend_gifs.csv", index=False, encoding="utf_8_sig")
        st.session_state.processing_complete = True
        
        # Store end times and calculate durations
        st.session_state.gif_end_time = datetime.now()
        st.session_state.end_time = datetime.now()
        
        # Calculate durations
        total_duration = st.session_state.end_time - st.session_state.start_time
        apify_duration = st.session_state.apify_end_time - st.session_state.apify_start_time
        gif_duration = st.session_state.gif_end_time - st.session_state.gif_start_time
        
        # Format durations
        def format_duration(duration):
            hours, remainder = divmod(duration.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        
        overall_progress.progress(100, text="All videos processed successfully!")
        st.success("Processing complete!")
        st.write(f"Process started at: {st.session_state.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        st.write(f"Process completed at: {st.session_state.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        st.write("Time breakdown:")
        st.write(f"- Total time: {format_duration(total_duration)}")
        st.write(f"- Apify task: {format_duration(apify_duration)}")
        st.write(f"- GIF conversion: {format_duration(gif_duration)}")

    except Exception as e:
        st.error(f"Error during GIF conversion: {str(e)}")
        # Reset session state on error
        st.session_state.processing_complete = False
        st.session_state.final_results = None
        st.session_state.gif_start_time = None
        st.session_state.gif_end_time = None

# Show download button if processing is complete
if st.session_state.processing_complete and st.session_state.final_results is not None:
    with open(f"{st.session_state.country_name}_trend_gifs.csv", "rb") as file:
        st.download_button(
            label="Download CSV File",
            data=file,
            file_name=f"{st.session_state.country_name}_trend_gifs.csv",
            mime="text/csv"
        )

# If Apify task failed, show error
elif st.session_state.apify_status == 'FAILED':
    st.error("Apify task failed. Please try again.")
    # Reset session state
    st.session_state.apify_status = None
    st.session_state.run_id = None
    st.session_state.dataset_id = None
    st.session_state.processing_complete = False
    st.session_state.final_results = None
    st.session_state.start_time = None
    st.session_state.end_time = None
    st.session_state.apify_start_time = None
    st.session_state.apify_end_time = None
    st.session_state.gif_start_time = None
    st.session_state.gif_end_time = None