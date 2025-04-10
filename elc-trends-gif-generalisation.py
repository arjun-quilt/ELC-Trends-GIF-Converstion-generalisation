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
import yt_dlp
import sys
import os
import gc
import time
import tempfile
import re
from playwright.async_api import async_playwright
import nest_asyncio
import subprocess
# Run the Playwright install command
subprocess.run(["playwright", "install-deps"], check=True)

# Initialize session state
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'processed_urls' not in st.session_state:
    st.session_state.processed_urls = {}
if 'current_batch' not in st.session_state:
    st.session_state.current_batch = 0

def reset_application():
    """Thoroughly reset the application state and perform cleanup"""
    try:
        # Show cleanup status
        status = st.empty()
        status.info("Cleaning up application state...")
        
        # Clear all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        # Clear all cached functions
        st.cache_data.clear()
        st.cache_resource.clear()
        
        # Clear any temporary files
        try:
            for file in os.listdir('.'):
                if file.endswith(('.mp4', '.gif', '.csv')):
                    try:
                        os.remove(file)
                        print(f"Removed temporary file: {file}")
                    except Exception as e:
                        print(f"Could not remove temporary file {file}: {str(e)}")
        except Exception as e:
            print(f"Error cleaning temporary files: {str(e)}")
        
        # Remove credentials file if it exists
        if 'temp_file_path' in globals() and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                print("Removed credentials file")
            except Exception as e:
                print(f"Could not remove credentials file: {str(e)}")
        
        # Force garbage collection
        gc.collect()
        
        # Show final message
        status.success("Cleanup complete! You can continue using the application.")
        
    except Exception as e:
        st.error(f"Error during cleanup: {str(e)}")
        # Log the error for debugging
        print(f"Error during reset_application: {str(e)}")
        
# Add reset button at the top
col1, col2 = st.columns([3, 1])
with col1:
    st.title("TikTok Video to GIF Converter")
with col2:
    if st.button("Reset Application", key="reset_button"):
        reset_application()

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

@st.cache_resource
def get_storage_client():
    return storage.Client()

@st.cache_data
def convert_to_gif(media_file, video_id, max_duration=2, fps=3):
    temp_gif_path = f"{video_id}.gif"
    with VideoFileClip(media_file) as clip:
        if clip.duration > max_duration:
            clip = clip.subclip(0, max_duration)
        clip = clip.set_fps(fps)
        clip.write_gif(temp_gif_path)
    return temp_gif_path

@st.cache_data
def upload_gif_to_gcs(local_file_path: str, video_id: str, bucket_name: str, gcs_folder: str):
    client = get_storage_client()
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

@st.cache_data
def yt_shorts_downloader(urls, bucket_name):
    # Ensure URLs is a list
    if not isinstance(urls, list):
        raise ValueError("The URLs parameter should be a list of strings.")

    # Initialize Google Cloud Storage client
    storage_client = get_storage_client()
    bucket = storage_client.bucket(bucket_name)

    # Create a single progress bar for all YouTube Shorts
    total_shorts = len(urls)
    st.write(f"Processing {total_shorts} YouTube Shorts...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    max_retries = 3
    retry_delay = 5  # seconds
    processed_urls = {}

    # Iterate through the list of URLs
    for idx, url in enumerate(urls):
        # Skip if URL is NaN or empty
        if pd.isna(url) or not url:
            processed_urls[url] = None
            continue

        # Update progress and status
        progress = (idx) / total_shorts
        progress_bar.progress(progress)
        status_text.text(f"Processing YouTube Short {idx + 1}/{total_shorts}: {url}")

        # Set options for yt-dlp
        ydl_opts = {
            'format': 'mp4',
            'outtmpl': '%(id)s.%(ext)s',  # Save as <video_id>.mp4
            'quiet': False,
            'socket_timeout': 30,
        }

        for retry in range(max_retries):
            try:
                # First get video ID
                with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                    info = ydl.extract_info(url, download=False)
                    video_id = info['id']
                    temp_file = f"{video_id}.mp4"

                # Download the video to temporary file
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                
                # Verify the downloaded file
                if not os.path.exists(temp_file):
                    raise Exception("Downloaded file does not exist")
                
                file_size = os.path.getsize(temp_file)
                if file_size == 0:
                    raise Exception("Downloaded file is empty")
                
                # Upload to GCS
                destination_blob = f"{video_id}.mp4"
                blob = bucket.blob(destination_blob)
                
                # Upload the file
                with open(temp_file, 'rb') as file:
                    blob.upload_from_file(file, content_type="video/mp4")
                
                # Clean up temporary file
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                
                gcs_url = f"https://storage.googleapis.com/{bucket_name}/{destination_blob}"
                processed_urls[url] = gcs_url
                break  # Success, break the retry loop
                
            except Exception as e:
                if retry < max_retries - 1:
                    time.sleep(retry_delay)
                    # Clean up any temporary files
                    if 'temp_file' in locals() and os.path.exists(temp_file):
                        os.unlink(temp_file)
                else:
                    processed_urls[url] = None
                    # Clean up any temporary files
                    if 'temp_file' in locals() and os.path.exists(temp_file):
                        os.unlink(temp_file)
                continue

    # Update final progress
    progress_bar.progress(1.0)
    status_text.text(f"Completed processing {total_shorts} YouTube Shorts")
    return processed_urls

# Apply nested event loop fix for Jupyter/Colab
nest_asyncio.apply()
@st.cache_data
def extract_and_download_douyin_video(video_page_url):
    async def _async_extract_and_download():
        match = re.search(r'/video/(\d+)', video_page_url)
        if not match:
            print("[ERROR] Unable to extract video ID from URL.")
            return None
        video_id = match.group(1)
        filename = f"{video_id}.mp4"

        print("[INFO] Starting Playwright session...")
        async with async_playwright() as p:
            print("[INFO] Launching browser...")
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            video_url = None

            async def intercept_response(response):
                nonlocal video_url
                url = response.url
                if "video" in url and (url.endswith(".mp4") or "mime_type=video_mp4" in url):
                    print(f"[INFO] Found video URL: {url}")
                    video_url = url

            page.on("response", intercept_response)

            print(f"[INFO] Navigating to {video_page_url}...")
            await page.goto(video_page_url, timeout=60000)

            print("[INFO] Waiting for video element to appear...")
            try:
                await page.wait_for_selector("video", timeout=15000)
            except Exception as e:
                print(f"[WARNING] Video element not found: {e}")

            print("[INFO] Waiting for network requests to be captured...")
            await page.wait_for_timeout(5000)
            await browser.close()

            if video_url:
                print(f"[SUCCESS] Extracted video URL: {video_url}")
                print(f"[INFO] Downloading video from {video_url}...")
                headers = {
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
                    "Referer": "https://www.douyin.com/",
                    "Range": "bytes=0-"
                }
                response = requests.get(video_url, headers=headers, stream=True)
                if response.status_code in [200, 206]:
                    print("[INFO] Saving video to file...")
                    with open(filename, "wb") as file:
                        for chunk in response.iter_content(chunk_size=1024):
                            file.write(chunk)
                    print(f"[SUCCESS] Video downloaded successfully as {filename}")
                    return filename  # Return the filename for further processing
                else:
                    print(f"[ERROR] Failed to download video. Status Code: {response.status_code}")
            else:
                print("[ERROR] Failed to extract video URL.")
            return None

    # Run async logic in already-running event loop
    return asyncio.get_event_loop().run_until_complete(_async_extract_and_download())

# Streamlit UI
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
country_name = st.text_input("Enter the country name (sheet name)")

if st.button("Process"):
    if uploaded_file and country_name:
        # Reset processing state
        st.session_state.processing = True
        st.session_state.processed_urls = {}
        st.session_state.current_batch = 0
        
        input_list_of_dicts = pd.read_excel(uploaded_file, sheet_name=country_name).to_dict(orient="records")
        
        # Separate URLs by type
        tiktok_urls = []
        gcs_urls = []
        youtube_urls = []
        douyin_urls = []
        for row in input_list_of_dicts:
            url = row["Links"]
            if pd.isna(url) or not url:
                continue
            if "tiktok.com" in url:
                tiktok_urls.append(url)
            elif "storage.googleapis.com" in url:
                gcs_urls.append(url)
            elif "youtube.com/shorts" in url or "youtu.be" in url:
                youtube_urls.append(url)
            elif "douyin.com" in url:
                douyin_urls.append(url)
        
        # Debug information
        st.write(f"Found {len(tiktok_urls)} TikTok URLs, {len(youtube_urls)} YouTube URLs, {len(gcs_urls)} GCS URLs, and {len(douyin_urls)} Douyin URLs")
        
        # Process TikTok videos if any
        if tiktok_urls:
            st.write(f"Processing {len(tiktok_urls)} TikTok videos...")
            success, dataset_ids = asyncio.run(process_tiktok_videos(tiktok_urls))
            if not success:
                st.error("Apify task failed. Please try again.")
                st.session_state.processing = False
                st.stop()
                
            st.write("All TikTok videos processed successfully. Fetching dataset items...")
            
            # Fetch items from all datasets
            all_items = []
            for dataset_id in dataset_ids:
                # st.write(f"Fetching items from dataset {dataset_id}...")
                dataset = asyncio.run(get_items(dataset_id))
                all_items.extend(dataset)
            
            st.write(f"Fetched {len(all_items)} items from all datasets.")
            
            # Create mapping for TikTok videos
            all_items_dict = {}
            for raw_row in all_items:
                try:
                    original_url = raw_row["submittedVideoUrl"]
                    all_items_dict[original_url] = {
                        "Gcs Url": raw_row["gcsMediaUrls"][0]
                    }
                except Exception as e:
                    # st.error(f"Error processing row: {raw_row} - {e}")
                    print(e)

            # Update GCS URLs for TikTok videos
            for input_dict in input_list_of_dicts:
                if "tiktok.com" in input_dict["Links"]:
                    clean_url = input_dict["Links"]
                    video_id = clean_url.split("?")[0].split("/")[-1]
                    input_dict["Gcs Url"] = f"https://storage.googleapis.com/tiktok-actor-content/{video_id}.mp4"
        
        # Process YouTube Shorts if any
        if youtube_urls:
            st.write(f"Processing {len(youtube_urls)} YouTube Shorts...")
            bucket_name = "tiktok-actor-content"
            processed_urls = yt_shorts_downloader(youtube_urls, bucket_name)
            
            # Update GCS URLs for YouTube videos
            for input_dict in input_list_of_dicts:
                if "youtube.com/shorts" in input_dict["Links"] or "youtu.be" in input_dict["Links"]:
                    if processed_urls.get(input_dict["Links"]):
                        input_dict["Gcs Url"] = processed_urls[input_dict["Links"]]
        
        # Process GCS videos if any
        if gcs_urls:
            st.write(f"Processing {len(gcs_urls)} GCS videos...")
            for input_dict in input_list_of_dicts:
                if "storage.googleapis.com" in input_dict["Links"]:
                    input_dict["Gcs Url"] = input_dict["Links"]

        # Process Douyin videos if any
        if douyin_urls:
            st.write(f"Processing {len(douyin_urls)} Douyin videos...")
            processed_douyin_urls = {}
            bucket_name = "tiktok-actor-content"
            client = get_storage_client()  # Get the storage client
            bucket = client.bucket(bucket_name)

            # Create a progress bar for Douyin videos
            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx, url in enumerate(douyin_urls):
                video_file = extract_and_download_douyin_video(url)
                if video_file:
                    # Upload to GCS
                    destination_blob = f"{os.path.basename(video_file)}"
                    blob = bucket.blob(destination_blob)
                    with open(video_file, 'rb') as file:
                        blob.upload_from_file(file, content_type="video/mp4")
                    gcs_url = f"https://storage.googleapis.com/{bucket_name}/{destination_blob}"
                    processed_douyin_urls[url] = gcs_url
                    os.unlink(video_file)  # Clean up the temporary file
                else:
                    processed_douyin_urls[url] = None

                # Update progress bar
                progress = (idx + 1) / len(douyin_urls)
                progress_bar.progress(progress)
                status_text.text(f"Processed {idx + 1}/{len(douyin_urls)} Douyin videos.")

            # Update GCS URLs for Douyin videos
            for input_dict in input_list_of_dicts:
                if "douyin.com" in input_dict["Links"]:
                    if processed_douyin_urls.get(input_dict["Links"]):
                        input_dict["Gcs Url"] = processed_douyin_urls[input_dict["Links"]]

        # Save intermediate results
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
                # Skip if GCS URL is NaN or empty
                if pd.isna(gcs_url) or not gcs_url:
                    continue

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
                # Clean up temporary files if they exist
                if 'temp_video_path' in locals() and os.path.exists(temp_video_path):
                    os.unlink(temp_video_path)
                if 'gif_path' in locals() and os.path.exists(gif_path):
                    os.unlink(gif_path)

        # Store final results
        final_results = pd.DataFrame(list_of_dicts)
        final_results.to_csv(f"{country_name}_trend_gifs.csv", index=False, encoding="utf_8_sig")
        
        overall_progress.progress(100, text="All videos processed successfully!")
        st.success("Processing complete! Check the generated CSV file for GIF URLs.")

        # Display sample results and statistics
        st.write("\n📊 Sample of Processed Results:")
        st.write("Here are the first 3 rows of your processed data:")
        
        # Create a sample dataframe with just the important columns
        sample_df = final_results[['Links', 'Gcs Url', 'GIF']].head(3)
        
        # Display the sample
        st.dataframe(sample_df)
        
        # Show processing statistics
        st.write("\n📈 Processing Statistics:")
        total_rows = len(final_results)
        rows_with_gcs = sum(final_results['Gcs Url'].notna())
        rows_with_gif = sum(final_results['GIF'].notna())
        
        st.write(f"Total rows processed: {total_rows}")
        st.write(f"Rows with GCS URLs: {rows_with_gcs} ({(rows_with_gcs/total_rows)*100:.1f}%)")
        st.write(f"Rows with GIF URLs: {rows_with_gif} ({(rows_with_gif/total_rows)*100:.1f}%)")

        # Show download button
        with open(f"{country_name}_trend_gifs.csv", "rb") as file:
            st.download_button(
                label="Download CSV File",
                data=file,
                file_name=f"{country_name}_trend_gifs.csv",
                mime="text/csv"
            )
        
        # Reset processing state
        st.session_state.processing = False