import yt_dlp
import pandas as pd
import os
import requests
import time
import shutil
from moviepy.editor import VideoFileClip
import subprocess
from yt_dlp.utils import DownloadError
from google.cloud import storage
import streamlit as st  
import tempfile
import io 
import imageio_ffmpeg as iio_ffmpeg
import base64
import tracemalloc
import gc
import aiohttp
import asyncio
from typing import List
import psutil

# Initialize memory tracking
tracemalloc.start()

# Add this at the start of your code, after imports
memory_placeholder = st.empty()  # Global placeholder for memory usage

# Add these constants at the top after imports
CLOUD_BATCH_SIZE = 3  # Smaller batch size for cloud processing
MEMORY_LIMIT_MB = 700  # Memory limit for cloud environment
CLEANUP_FREQUENCY = 2  # Cleanup after every 2 videos

############## Helper functions starts #############

async def run_actor_task(data: dict):
    """Async version of run_actor_task"""
    headers = {"Content-Type": "application/json"}
    url = f"https://api.apify.com/v2/actor-tasks/H70fR5ndjUD0loq5H/runs?token=apify_api_VUQNA5xFO4IwieTeWX7HmKUYnNZOnw0c2tgk"
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data, headers=headers) as response:
            return await response.json()

async def check_run_status(run_id: str, api_token: str, status_placeholder) -> bool:
    """Check status of a single Apify run"""
    url = f"https://api.apify.com/v2/actor-runs/{run_id}?token={api_token}"
    max_retries = 3
    retry_delay = 5

    async with aiohttp.ClientSession() as session:
        while True:
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        run_data = await response.json()
                        status = run_data.get('data', {}).get('status')
                        
                        if status == 'SUCCEEDED':
                            status_placeholder.success(f"Run {run_id}: SUCCEEDED")
                            return True
                        elif status == 'FAILED':
                            status_placeholder.error(f"Run {run_id}: FAILED")
                            return False
                        elif status == 'RUNNING':
                            status_placeholder.info(f"Run {run_id}: Still running...")
                            await asyncio.sleep(5)
                        else:
                            status_placeholder.warning(f"Run {run_id}: Unknown status - {status}")
                            return False
                    else:
                        raise Exception(f"HTTP {response.status}")

            except Exception as e:
                status_placeholder.warning(f"Run {run_id}: Request failed - {str(e)}")
                await asyncio.sleep(retry_delay)

async def process_tiktok_videos(tiktok_videos: List[str], batch_size: int = 20):
    """Process all TikTok videos in parallel batches"""
    if not tiktok_videos:
        return True

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
            run_ids = [response['data']['id'] for response in responses]
            st.write(f"Started {len(run_ids)} Apify runs")

            # Check status of all runs
            status_tasks = []
            API_TOKEN = "apify_api_VUQNA5xFO4IwieTeWX7HmKUYnNZOnw0c2tgk"
            for run_id in run_ids:
                status_tasks.append(check_run_status(run_id, API_TOKEN, status_placeholder))
            
            with st.spinner("Waiting for all TikTok video downloads to complete..."):
                results = await asyncio.gather(*status_tasks)
                
                if all(results):
                    st.success("All TikTok videos processed successfully!")
                    return True
                else:
                    failed_runs = sum(1 for r in results if not r)
                    st.error(f"Failed to process {failed_runs} batches of TikTok videos")
                    return False
                    
        except Exception as e:
            st.error(f"Error processing TikTok videos: {str(e)}")
            return False

async def upload_to_gcs_async(bucket, video_data, destination_blob):
    """Async version of upload_to_gcs"""
    loop = asyncio.get_event_loop()
    blob = bucket.blob(destination_blob)
    await loop.run_in_executor(None, lambda: blob.upload_from_file(video_data, content_type="video/mp4"))
    print(f"Video uploaded to {destination_blob} in bucket {bucket.name}.")

async def check_apify_run_status_async(run_id, api_token):
    """Async version of check_apify_run_status"""
    url = f"https://api.apify.com/v2/actor-runs/{run_id}?token={api_token}"
    
    # Streamlit component to display status
    status_message = st.empty()  # Create an empty placeholder for status updates
    max_retries = 3
    retry_delay = 5
    timeout = 30  # seconds

    async with aiohttp.ClientSession() as session:
        while True:
            for retry in range(max_retries):
                try:
                    async with session.get(url, timeout=timeout) as response:
                        if response.status == 200:
                            run_data = await response.json()
                            status = run_data.get('data', {}).get('status')

                            if status == 'SUCCEEDED':
                                status_message.success("SUCCEEDED")
                                return True
                            elif status == 'FAILED':
                                status_message.error("FAILED")
                                return False
                            elif status == 'RUNNING':
                                status_message.info("Run is still in progress...")
                                break  # Break retry loop on successful check
                            else:
                                status_message.warning(f"Unknown status: {status}")
                                break  # Break retry loop on successful check
                        else:
                            if retry < max_retries - 1:
                                status_message.warning(f"Request failed (attempt {retry + 1}/{max_retries}). Retrying...")
                                await asyncio.sleep(retry_delay)
                            else:
                                status_message.error(f"Error fetching run status: {response.status}, {await response.text()}")
                                return False

                except asyncio.TimeoutError:
                    if retry < max_retries - 1:
                        status_message.warning(f"Request timed out (attempt {retry + 1}/{max_retries}). Retrying...")
                        await asyncio.sleep(retry_delay)
                    else:
                        status_message.error("Failed to check status due to timeout after multiple retries")
                        return False
                except Exception as e:
                    if retry < max_retries - 1:
                        status_message.warning(f"Request failed (attempt {retry + 1}/{max_retries}): {str(e)}. Retrying...")
                        await asyncio.sleep(retry_delay)
                    else:
                        status_message.error(f"Failed to check status after multiple retries: {str(e)}")
                        return False

            # Wait before next status check
            await asyncio.sleep(5)

async def yt_shorts_downloader_async(urls, bucket_name):
    """Async version of yt_shorts_downloader with cloud optimizations"""
    if not isinstance(urls, list):
        raise ValueError("The URLs parameter should be a list of strings.")

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    progress_bar = st.progress(0)
    status_text = st.empty()
    total_shorts = len(urls)
    max_retries = 3
    retry_delay = 5

    # Reduce concurrent downloads for cloud environment
    semaphore = asyncio.Semaphore(2)  # Limit to 2 concurrent downloads

    async def download_single_short(url, idx):
        async with semaphore:
            # Check memory before each download
            check_memory_limit()
            
            progress = (idx + 1) / total_shorts
            progress_bar.progress(progress)
            status_text.text(f"Processing YouTube Short {idx + 1}/{total_shorts}")

            ydl_opts = {
                'format': 'mp4',
                'outtmpl': '-',
                'quiet': True,
                'socket_timeout': 30,
            }

            for retry in range(max_retries):
                try:
                    loop = asyncio.get_event_loop()
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        result = await loop.run_in_executor(None, lambda: ydl.extract_info(url, download=False))
                        video_id = result['id']
                        video_content = await loop.run_in_executor(None, lambda: ydl.urlopen(result['url']).read())
                        video_data = io.BytesIO(video_content)
                        destination_blob = f"{video_id}.mp4"
                        await upload_to_gcs_async(bucket, video_data, destination_blob)
                        print(f"Uploaded {destination_blob} to bucket {bucket_name}.")
                        
                        # Cleanup after each successful upload
                        del video_content
                        del video_data
                        gc.collect()
                        break
                except Exception as e:
                    if retry < max_retries - 1:
                        st.warning(f"Attempt {retry + 1} failed for {url}. Retrying in {retry_delay} seconds...")
                        await asyncio.sleep(retry_delay)
                    else:
                        st.error(f"Failed to process YouTube Short {url} after {max_retries} attempts: {str(e)}")
                    continue

    # Process videos in smaller batches
    batch_size = CLOUD_BATCH_SIZE
    for i in range(0, len(urls), batch_size):
        batch = urls[i:i + batch_size]
        tasks = [download_single_short(url, idx + i) for idx, url in enumerate(batch)]
        await asyncio.gather(*tasks)
        
        # Add delay between batches
        await asyncio.sleep(2)

    progress_bar.progress(1.0)
    status_text.text("Completed processing all YouTube Shorts!")

def log_memory_usage():
    """Monitor memory usage with updating display"""
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_placeholder.text(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")

async def download_and_trim_video_async(url, output_dir=os.path.join(os.getcwd(), 'videos'), duration=10):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video_filename = os.path.basename(url)
    output_path = os.path.join(output_dir, video_filename)
    trimmed_output = os.path.join(output_dir, f"trimmed_{video_filename}")

    try:
        ydl_opts = {
            'format': 'mp4',
            'outtmpl': output_path,
        }

        # Run yt-dlp download in thread pool
        loop = asyncio.get_event_loop()
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            await loop.run_in_executor(None, lambda: ydl.download([url]))

        # Add explicit cleanup for video processing
        if os.path.exists(output_path):
            clip = await loop.run_in_executor(None, lambda: VideoFileClip(output_path))
            await loop.run_in_executor(None, lambda: clip.close())
            del clip
            gc.collect()

        ffmpeg_path = iio_ffmpeg.get_ffmpeg_exe()
        command = [
            ffmpeg_path,
            '-i', output_path,
            '-t', str(duration),
            '-c', 'copy',
            '-y',
            trimmed_output
        ]
        
        # Run ffmpeg command in thread pool
        result = await loop.run_in_executor(None, lambda: subprocess.run(command, capture_output=True, text=True, env=os.environ))

        if result.returncode != 0:
            print(f"FFmpeg command failed with error: {result.stderr}")
            return None

        # Clean up files
        await loop.run_in_executor(None, lambda: os.remove(output_path))
        await loop.run_in_executor(None, lambda: os.rename(trimmed_output, output_path))
        return output_path

    except Exception as e:
        print(f"Failed to download/trim video: {e}")
        if 'clip' in locals():
            await loop.run_in_executor(None, lambda: clip.close())
            del clip
        gc.collect()
        return None
    finally:
        if os.path.exists(trimmed_output):
            await loop.run_in_executor(None, lambda: os.remove(trimmed_output))

async def upload_gif_to_gcs_async(bucket_name, gif_path):
    loop = asyncio.get_event_loop()
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    destination_blob = f"gifs_20240419/{os.path.basename(gif_path)}"

    # Upload file in thread pool
    blob = bucket.blob(destination_blob)
    await loop.run_in_executor(None, lambda: blob.upload_from_filename(gif_path, content_type="image/gif"))
    print(f"Uploaded {gif_path} to bucket {bucket_name} at {destination_blob}.")

async def convert_to_gif_async(media_file, max_duration=10, fps=10, output_dir=os.path.join(os.getcwd(), 'gifs')):
    """Async version of convert_to_gif"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    clip = None
    try:
        # Run video processing in a thread pool to not block the event loop
        loop = asyncio.get_event_loop()
        clip = await loop.run_in_executor(None, lambda: VideoFileClip(media_file))
        
        if clip.duration > max_duration:
            clip = await loop.run_in_executor(None, lambda: clip.subclip(0, max_duration))

        clip = await loop.run_in_executor(None, lambda: clip.set_fps(fps))
        output_gif_path = os.path.join(output_dir, os.path.splitext(os.path.basename(media_file))[0] + '.gif')
        
        # Run GIF writing in thread pool
        await loop.run_in_executor(None, lambda: clip.write_gif(output_gif_path))
        print(f"Converted {media_file} to {output_gif_path}")
        return output_gif_path
    except Exception as e:
        print(f"Failed to convert {media_file}: {e}")
        return None
    finally:
        if clip:
            await loop.run_in_executor(None, lambda: clip.close())
            del clip
        gc.collect()

async def resize_gif_async(input_gif, max_size_mb=1.99, processed_dir=os.path.join(os.getcwd(), 'processed_gifs')):
    """Async version of resize_gif"""
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    loop = asyncio.get_event_loop()
    clip = None
    try:
        # Load video in thread pool
        clip = await loop.run_in_executor(None, lambda: VideoFileClip(input_gif))
        fps = clip.fps
        current_size = os.path.getsize(input_gif)

        if current_size > max_size_mb * 1024 * 1024:
            reduction_factor = 0.95
            while current_size > max_size_mb * 1024 * 1024:
                new_duration = clip.duration * reduction_factor
                clip = await loop.run_in_executor(None, lambda: clip.subclip(0, new_duration))
                await loop.run_in_executor(None, lambda: clip.write_gif("temp_resized.gif", fps=fps))
                current_size = os.path.getsize("temp_resized.gif")
                os.remove("temp_resized.gif")
                reduction_factor *= 0.95

            base, ext = os.path.splitext(os.path.basename(input_gif))
            output_gif = os.path.join(processed_dir, f"{base}{ext}")
            await loop.run_in_executor(None, lambda: clip.write_gif(output_gif, fps=fps))
            print(f"Resized GIF saved as {output_gif}, size: {current_size/1024/1024:.2f} MB")
            return output_gif if current_size <= max_size_mb * 1024 * 1024 else None
        else:
            shutil.copy(input_gif, processed_dir)
            return input_gif
    finally:
        if clip:
            await loop.run_in_executor(None, lambda: clip.close())
            del clip
        gc.collect()

async def process_video_async(video_url, output_dir):
    """Process a single video asynchronously"""
    try:
        video_path = await download_and_trim_video_async(video_url)
        if video_path:
            try:
                gif_path = await convert_to_gif_async(video_path, output_dir=output_dir)
                if gif_path:
                    try:
                        resized_gif_path = await resize_gif_async(gif_path)
                        if resized_gif_path and os.path.exists(resized_gif_path):
                            await upload_gif_to_gcs_async('tiktok-actor-content', resized_gif_path)
                            await asyncio.get_event_loop().run_in_executor(None, lambda: os.remove(resized_gif_path))
                            return True
                    finally:
                        if os.path.exists(gif_path):
                            await asyncio.get_event_loop().run_in_executor(None, lambda: os.remove(gif_path))
            finally:
                if os.path.exists(video_path):
                    await asyncio.get_event_loop().run_in_executor(None, lambda: os.remove(video_path))
        return False
    except Exception as e:
        print(f"Error processing video {video_url}: {str(e)}")
        return False

# Add this function for memory monitoring
def check_memory_limit():
    """Check if memory usage is approaching limit"""
    process = psutil.Process()
    memory_info = process.memory_info()
    current_memory_mb = memory_info.rss / 1024 / 1024
    if current_memory_mb > MEMORY_LIMIT_MB:
        st.warning(f"Memory usage high ({current_memory_mb:.2f} MB). Triggering cleanup...")
        st.cache_data.clear()
        st.cache_resource.clear()
        gc.collect()
        return True
    return False

# Modify process_videos_batch_async to include memory checks
async def process_videos_batch_async(videos, output_dir, batch_size=CLOUD_BATCH_SIZE):
    """Process videos in smaller batches with memory monitoring"""
    results = []
    for i in range(0, len(videos), batch_size):
        # Check memory before each batch
        check_memory_limit()
        
        batch = videos[i:i + batch_size]
        tasks = [process_video_async(video_url, output_dir) for video_url in batch]
        try:
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
            
            # Cleanup after each batch
            st.cache_data.clear()
            st.cache_resource.clear()
            gc.collect()
            
            # Add a small delay between batches to allow system to stabilize
            await asyncio.sleep(2)
            
        except Exception as e:
            st.error(f"Error processing batch: {str(e)}")
            # Continue with next batch instead of failing completely
            continue
    return results

# Modify process_videos_from_excel to handle smaller chunks
def process_videos_from_excel(input_excel, sheet_name, output_dir=os.path.join(os.getcwd(), 'gifs')):
    try:
        st.cache_data.clear()
        st.cache_resource.clear()
        gc.collect()
        
        log_memory_usage()
        
        df = pd.read_excel(input_excel, sheet_name=sheet_name)
        valid_urls = df[~df['Gcs Url'].isna()]['Gcs Url'].tolist()
        
        # Process in smaller chunks of 15 videos
        chunk_size = 15
        chunks = [valid_urls[i:i + chunk_size] for i in range(0, len(valid_urls), chunk_size)]
        
        total_videos = len(valid_urls)
        processed_count = 0
        progress_bar = st.progress(0)
        counter_placeholder = st.empty()
        
        if total_videos == 0:
            st.warning("No valid videos to process.")
            return df

        # Process each chunk separately
        for chunk_index, chunk in enumerate(chunks):
            st.info(f"Processing chunk {chunk_index + 1}/{len(chunks)} ({len(chunk)} videos)")
            
            # Check memory before processing chunk
            check_memory_limit()
            
            # Process videos in current chunk
            for index, video_url in enumerate(chunk):
                try:
                    if processed_count > 0 and processed_count % CLEANUP_FREQUENCY == 0:
                        st.cache_data.clear()
                        st.cache_resource.clear()
                        gc.collect()
                        log_memory_usage()

                    if pd.isna(video_url) or not isinstance(video_url, str):
                        continue

                    print(f"Processing video URL: {video_url}")
                    
                    # Process video with proper cleanup
                    video_path = asyncio.run(download_and_trim_video_async(video_url))
                    if video_path:
                        try:
                            gif_path = asyncio.run(convert_to_gif_async(video_path, output_dir=output_dir))
                            if gif_path:
                                try:
                                    resized_gif_path = asyncio.run(resize_gif_async(gif_path))
                                    if resized_gif_path and os.path.exists(resized_gif_path):
                                        asyncio.run(upload_gif_to_gcs_async('tiktok-actor-content', resized_gif_path))
                                        os.remove(resized_gif_path)
                                        processed_count += 1
                                finally:
                                    if os.path.exists(gif_path):
                                        os.remove(gif_path)
                        finally:
                            if os.path.exists(video_path):
                                os.remove(video_path)

                    # Update progress
                    progress = min(processed_count / total_videos, 1.0)
                    progress_bar.progress(progress)
                    counter_placeholder.text(f"Processed {processed_count}/{total_videos} videos")
                    
                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")
                    continue
                
                # Check memory after each video
                check_memory_limit()
            
            # Save progress after each chunk
            output_file_name = 'updated_tiktok_urls.xlsx'
            df.to_excel(output_file_name, index=False)
            
            # Add delay between chunks
            time.sleep(3)

        progress_bar.progress(1.0)
        counter_placeholder.text(f"Completed processing all {processed_count}/{total_videos} videos!")
        
        st.success(f"All videos processed successfully! Processed {processed_count} out of {total_videos} videos.")
        return df
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return df
    finally:
        st.cache_data.clear()
        st.cache_resource.clear()
        gc.collect()
        log_memory_usage()

# Add this new async function for checking URLs
async def check_gcs_url_status(url, session):
    """Async function to check GCS URL status"""
    try:
        async with session.head(url, timeout=10) as response:
            return url, response.status == 200
    except Exception:
        return url, False

# Modify the process_videos_from_excel function to include async URL checking
async def check_all_gcs_urls(urls_to_check):
    """Check multiple GCS URLs in parallel"""
    async with aiohttp.ClientSession() as session:
        tasks = [check_gcs_url_status(url, session) for url in urls_to_check]
        results = await asyncio.gather(*tasks)
        return dict(results)

############### Helper functions ends ##############

# Streamlit app title
st.title("Elc Trends Gif Converstion")

# Extract the secret
gcp_secret = st.secrets["gcp_secret"]

# Write the secret to a temporary file
with tempfile.NamedTemporaryFile(delete=False, mode="w") as temp_file:
    temp_file.write(gcp_secret)
    temp_file_path = temp_file.name

# Set the environment variable to the temporary file path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_file_path

# Initialize Google Cloud Storage client
storage_client = storage.Client()
bucket = storage_client.bucket('tiktok-actor-content')

# File uploader for Excel and CSV files
input_excel = st.file_uploader("Upload Excel or CSV File", type=["xlsx", "csv"])  # Updated to accept CSV
st.write("Upload an Excel or CSV sheet in Format: ")
st.write("Column_Name: Links")
st.write("Tab_Name: Master_Sheet")

if input_excel:
    # Read the file based on its extension
    if input_excel.name.endswith('.xlsx'):
        df = pd.read_excel(input_excel, sheet_name='Master_Sheet')  # Default sheet name for Excel
    elif input_excel.name.endswith('.csv'):
        df = pd.read_csv(input_excel)  # Read CSV file
    st.write("Data from file:", df)
    st.write('length of input: ', len(df))

    # Create a copy of the dataframe to preserve all data
    processed_df = df.copy()

    # Get valid URLs (non-empty) from Links column
    valid_urls = df['Links'].dropna().tolist()  # Drop NaN values before converting to list

    # Classify URLs more precisely
    youtube_shorts = [url for url in valid_urls if isinstance(url, str) and 
                     ('youtube.com' in url.lower() or 'youtu.be' in url.lower())]

    gcs_urls = [url for url in valid_urls if isinstance(url, str) and 
                url.startswith("https://storage.googleapis.com/tiktok-actor-content/")]

    tiktok_videos = [url for url in valid_urls if isinstance(url, str) and 
                    'tiktok.com' in url.lower() and url.split("/")[-1].isdigit()]

    # Debug information
    st.write(f"Found {len(youtube_shorts)} YouTube Shorts")
    st.write(f"Found {len(gcs_urls)} GCS URLs")
    st.write(f"Found {len(tiktok_videos)} TikTok videos")

    # Button to start downloading and processing videos
    if st.button("Download and Process Videos"):
        # Clear caches before starting the download process
        st.cache_data.clear()
        st.cache_resource.clear()
        
        # Only process YouTube shorts if there are any
        if youtube_shorts:
            try:
                with st.spinner("Downloading YouTube Shorts..."):
                    asyncio.run(yt_shorts_downloader_async(
                        urls=youtube_shorts,
                        bucket_name="tiktok-actor-content"
                    ))
                    gc.collect()
            except Exception as e:
                print(f"An error occurred while downloading YouTube Shorts: {e}")
        
        # Process TikTok videos using async
        if tiktok_videos:
            success = asyncio.run(process_tiktok_videos(tiktok_videos))
            if not success:
                st.error("TikTok video processing failed")

        # Process each row and update GCS/GIF URLs
        for index, row in processed_df.iterrows():
            # Preserve the existing data column and other columns
            if pd.isna(row["Links"]) or not isinstance(row["Links"], str):
                # Keep existing data and URLs if they exist
                if "Gcs Url" not in processed_df.columns:
                    processed_df.at[index, "Gcs Url"] = None
                if "Gif Url" not in processed_df.columns:
                    processed_df.at[index, "Gif Url"] = None
            else:
                # Process valid links
                clean_url = row["Links"]
                # If it's a GCS URL, use it directly
                if clean_url.startswith("https://storage.googleapis.com/tiktok-actor-content/"):
                    processed_df.at[index, "Gcs Url"] = clean_url
                else:
                    # For TikTok/YouTube URLs, create new GCS URL
                    video_id = clean_url.split("?")[0].split("/")[-1]
                    processed_df.at[index, "Gcs Url"] = f"https://storage.googleapis.com/tiktok-actor-content/{video_id}.mp4"
                
                # Set Gif URL for all types
                video_id = processed_df.at[index, "Gcs Url"].split("/")[-1].replace(".mp4", "")
                processed_df.at[index, "Gif Url"] = f"https://storage.googleapis.com/tiktok-actor-content/gifs_20240419/{video_id}.gif"

        # Save the updated data to Excel file
        output_file_name = 'updated_tiktok_urls.xlsx'
        processed_df.to_excel(output_file_name, index=False)
        st.success(f"Updated Excel file saved as {output_file_name}")

        # Process videos to create GIFs
        with st.spinner("Converting videos to GIFs..."):
            final_df = process_videos_from_excel(output_file_name, 'Sheet1')
            final_df.to_excel(output_file_name, index=False)
        
        # Clear caches before file download
        st.cache_data.clear()
        st.cache_resource.clear()
        
        # Offer download of final results
        st.download_button(
            label="Download Updated Gif Urls Sheet",
            data=open(output_file_name, "rb").read(),
            file_name=output_file_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# Clean up temporary files and resources
gc.collect()