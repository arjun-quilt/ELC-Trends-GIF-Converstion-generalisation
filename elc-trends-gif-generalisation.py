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
from typing import List

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
    with VideoFileClip(media_file) as clip:
        if clip.duration > max_duration:
            clip = clip.subclip(0, max_duration)
        clip = clip.set_fps(fps)
        clip.write_gif(temp_gif_path)
    return temp_gif_path

# Function to upload GIF to GCS
def upload_gif_to_gcs(local_file_path: str, video_id: str, bucket_name: str, gcs_folder: str):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    gcs_blob_path = f"{gcs_folder}/{video_id}.gif"
    blob = bucket.blob(gcs_blob_path)
    blob.upload_from_filename(local_file_path)
    return f"https://storage.googleapis.com/{bucket_name}/{gcs_blob_path}"

async def check_run_status(run_id: str, api_token: str, status_placeholder) -> bool:
    """Check status of a single Apify run (asynchronous version)"""
    url = f"https://api.apify.com/v2/actor-runs/{run_id}?token={api_token}"
    retry_delay = 5

    while True:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                if response.status_code == 200:
                    run_data = response.json()
                    status = run_data.get('data', {}).get('status')
                    
                    if status == 'SUCCEEDED':
                        status_placeholder.success(f"Run {run_id}: SUCCEEDED")
                        return True
                    elif status == 'FAILED':
                        status_placeholder.error(f"Run {run_id}: FAILED")
                        return False
                    elif status == 'RUNNING':
                        status_placeholder.info(f"Run {run_id}: Still running... (Last checked: {time.strftime('%H:%M:%S')})")
                        await asyncio.sleep(5)
                    else:
                        status_placeholder.warning(f"Run {run_id}: Unknown status - {status}")
                        return False
                else:
                    raise Exception(f"HTTP {response.status_code}")

        except Exception as e:
            status_placeholder.warning(f"Run {run_id}: Request failed - {str(e)}")
            await asyncio.sleep(retry_delay)

async def process_tiktok_videos(tiktok_videos: List[str], batch_size: int = 5):
    """Process all TikTok videos in parallel batches"""
    if not tiktok_videos:
        return True, None

    # Split videos into batches
    batches = [tiktok_videos[i:i + batch_size] for i in range(0, len(tiktok_videos), batch_size)]
    
    status_placeholder = st.empty()
    
    # Start all batches
    with st.spinner(f"Processing {len(tiktok_videos)} TikTok videos in {len(batches)} batches..."):
        try:
            # Launch all actor tasks
            tasks = []
            for batch in batches:
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
                    "postURLs": batch
                }
                tasks.append(run_actor_task(input_params))
            
            responses = await asyncio.gather(*tasks)
            run_ids = [response.json()['data']['id'] for response in responses]
            st.write(f"Started {len(run_ids)} Apify runs")

            # Check status of all runs
            status_tasks = []
            API_TOKEN = "apify_api_VUQNA5xFO4IwieTeWX7HmKUYnNZOnw0c2tgk"
            for run_id in run_ids:
                status_tasks.append(check_run_status(run_id, API_TOKEN, status_placeholder))
            
            # Wait for all status checks to complete
            results = await asyncio.gather(*status_tasks)
            
            # Check if all runs succeeded
            if all(results):
                st.success("All Apify runs completed successfully!")
                # Get dataset IDs from all successful runs
                dataset_ids = [response.json()["data"]["defaultDatasetId"] for response in responses]
                return True, dataset_ids
            else:
                st.error("Some Apify runs failed. Please check the logs above.")
                return False, None
                
        except Exception as e:
            st.error(f"Error during batch processing: {str(e)}")
            return False, None

# Streamlit UI
st.title("TikTok Video to GIF Converter")

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
country_name = st.text_input("Enter the country name (sheet name)")

if st.button("Process"):
    if uploaded_file and country_name:
        input_list_of_dicts = pd.read_excel(uploaded_file, sheet_name=country_name).to_dict(orient="records")
        
        # Extract video URLs
        video_urls = [row["Links"] for row in input_list_of_dicts]
        
        # Process videos in batches
        success, dataset_ids = asyncio.run(process_tiktok_videos(video_urls))
        if success:
            st.write("All videos processed successfully. Fetching dataset items...")
            
            # Fetch items from all datasets
            all_items = []
            for dataset_id in dataset_ids:
                st.write(f"Fetching items from dataset {dataset_id}...")
                dataset = asyncio.run(get_items(dataset_id))
                all_items.extend(dataset)
            
            st.write(f"Fetched {len(all_items)} items from all datasets.")
            
            all_items_dict = {}
            for raw_row in all_items:
                try:
                    original_url = raw_row["submittedVideoUrl"]
                    all_items_dict[original_url] = {
                        "Gcs Url": raw_row["gcsMediaUrls"][0]
                    }
                except Exception as e:
                    st.error(f"Error processing row: {raw_row} - {e}")

            for input_dict in input_list_of_dicts:
                clean_url = input_dict["Links"]
                video_id = clean_url.split("?")[0].split("/")[-1]
                input_dict["Gcs Url"] = f"https://storage.googleapis.com/tiktok-actor-content/{video_id}.mp4"

            output_df = pd.DataFrame(input_list_of_dicts)
            output_df.to_csv(f"{country_name}_duration.csv", index=False, encoding="utf_8_sig")
            st.write(f"Generated CSV file: {country_name}_duration.csv")

            # Download videos and convert to GIFs
            df = pd.read_csv(f"{country_name}_duration.csv")
            list_of_dicts = df.to_dict(orient="records")

            # Initialize overall progress bar
            total_videos = len(list_of_dicts)
            overall_progress = st.progress(0, text=f"Processing {total_videos} videos...")
            
            for index, raw_row in enumerate(list_of_dicts, 1):
                gcs_url = raw_row["Gcs Url"]
                try:
                    video_id = gcs_url.split("/")[-1].split(".")[0]
                    
                    # Create a temporary file for the video
                    temp_video_path = f"{video_id}.mp4"
                    urllib.request.urlretrieve(gcs_url, temp_video_path)
                    
                    gif_path = convert_to_gif(temp_video_path, video_id)
                    
                    # Upload GIF to GCS
                    bucket_name = "tiktok-actor-content"
                    gcs_folder = "gifs_20240419"
                    
                    gif_url = upload_gif_to_gcs(gif_path, video_id, bucket_name, gcs_folder)
                    
                    # Clean up temporary files
                    os.unlink(temp_video_path)
                    os.unlink(gif_path)
                    
                    raw_row["GIF"] = gif_url
                    
                    # Update overall progress
                    progress_percent = int((index / total_videos) * 100)
                    overall_progress.progress(progress_percent, text=f"Processed {index}/{total_videos} videos")
                    
                except Exception as e:
                    st.error(f"Error processing video: {gcs_url} - {e}")

            # Store final results
            final_results = pd.DataFrame(list_of_dicts)
            final_results.to_csv(f"{country_name}_trend_gifs.csv", index=False, encoding="utf_8_sig")
            
            overall_progress.progress(100, text="All videos processed successfully!")
            st.success("Processing complete! Check the generated CSV file for GIF URLs.")

            # Show download button
            with open(f"{country_name}_trend_gifs.csv", "rb") as file:
                st.download_button(
                    label="Download CSV File",
                    data=file,
                    file_name=f"{country_name}_trend_gifs.csv",
                    mime="text/csv"
                )
        else:
            st.error("Apify task failed. Please try again.")