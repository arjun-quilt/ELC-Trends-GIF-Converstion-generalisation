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

# Initialize memory tracking
tracemalloc.start()

############## Helper functions starts #############

def yt_shorts_downloader(urls, bucket_name):
    # Ensure URLs is a list
    if not isinstance(urls, list):
        raise ValueError("The URLs parameter should be a list of strings.")

    # Initialize Google Cloud Storage client
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Create progress bar and counter for YouTube Shorts
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_shorts = len(urls)
    max_retries = 3
    retry_delay = 5  # seconds

    # Iterate through the list of URLs
    for idx, url in enumerate(urls):
        # Update progress
        progress = (idx + 1) / total_shorts
        progress_bar.progress(progress)
        status_text.text(f"Processing YouTube Short {idx + 1}/{total_shorts}")

        # Set options for yt-dlp
        ydl_opts = {
            'format': 'mp4',  # Specify MP4 format
            'outtmpl': '-',   # Output to stdout (streaming)
            'quiet': True,    # Suppress yt-dlp's output
            'socket_timeout': 30,  # Increase socket timeout
        }

        for retry in range(max_retries):
            try:
                # Download video and upload to GCS
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    result = ydl.extract_info(url, download=False)
                    video_id = result['id']

                    # Get the actual video content
                    video_content = ydl.urlopen(result['url']).read()
                    video_data = io.BytesIO(video_content)

                    # Define the destination file name in the bucket
                    destination_blob = f"{video_id}.mp4"

                    # Create a blob in the bucket and upload the video
                    blob = bucket.blob(destination_blob)
                    blob.upload_from_file(video_data, content_type="video/mp4")

                    print(f"Uploaded {destination_blob} to bucket {bucket_name}.")
                    break  # Success, break the retry loop
            except Exception as e:
                if retry < max_retries - 1:
                    st.warning(f"Attempt {retry + 1} failed for {url}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    st.error(f"Failed to process YouTube Short {url} after {max_retries} attempts: {str(e)}")
                continue

    # Complete the progress bar
    progress_bar.progress(1.0)
    status_text.text("Completed processing all YouTube Shorts!")

# Helper function to upload to GCS (if needed externally)
def upload_to_gcs(bucket, video_data, destination_blob):
    blob = bucket.blob(destination_blob)
    blob.upload_from_file(video_data, content_type="video/mp4")
    print(f"Video uploaded to {destination_blob} in bucket {bucket.name}.")

# Function to convert media files to GIFs
def convert_to_gif(media_file, max_duration=10, fps=10, output_dir=os.path.join(os.getcwd(), 'gifs')):
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

def upload_gif_to_gcs(bucket_name, gif_path):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    destination_blob = f"gifs_20240419/{os.path.basename(gif_path)}"  # Destination path in GCS

    blob = bucket.blob(destination_blob)
    blob.upload_from_filename(gif_path, content_type="image/gif")
    print(f"Uploaded {gif_path} to bucket {bucket_name} at {destination_blob}.")

# Function to download video from GCS URL
def download_and_trim_video(url, output_dir=os.path.join(os.getcwd(), 'videos'), duration=10):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video_filename = os.path.basename(url)
    output_path = os.path.join(output_dir, video_filename)

    ydl_opts = {
        'format': 'mp4',
        'outtmpl': output_path,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        print(f"Failed to download video: {e}")
        #st.error(f"Failed to download video from URL: {url} - Error: {e}")  # Display the video URL and error message in Streamlit        return None

    trimmed_output = os.path.join(output_dir, f"trimmed_{video_filename}")

    try:
        ffmpeg_path = iio_ffmpeg.get_ffmpeg_exe()
        print(f"FFmpeg path: {ffmpeg_path}")
        command = [
            ffmpeg_path,
            '-i', output_path,
            '-t', str(duration),
            '-c', 'copy',
            '-y',
            trimmed_output
        ]
        result = subprocess.run(command, capture_output=True, text=True, env=os.environ)

        if result.returncode != 0:
            print(f"FFmpeg command failed with error: {result.stderr}")
            return None

        os.remove(output_path)
        os.rename(trimmed_output, output_path)
        return output_path

    except Exception as e:
        print(f"Failed to trim the video: {e}")
        return None

def resize_gif(input_gif, max_size_mb=1.99, processed_dir=os.path.join(os.getcwd(), 'processed_gifs')):
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    clip = VideoFileClip(input_gif)
    try:
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
            return output_gif if current_size <= max_size_mb * 1024 * 1024 else None  # Return None if still too large
        else:
            # If not resizing, just copy the original GIF to the processed directory
            shutil.copy(input_gif, processed_dir)
            return input_gif  # Return the original GIF path
    finally:
        clip.close()  # Close the VideoFileClip object
        del clip  # Delete the VideoFileClip object
        gc.collect()  # Collect garbage to free up memory

#call this function at last
def process_videos_from_excel(input_excel, sheet_name, output_dir=os.path.join(os.getcwd(), 'gifs')):
    # Clear caches before starting the process
    st.cache_data.clear()
    st.cache_resource.clear()
    
    df = pd.read_excel(input_excel, sheet_name=sheet_name)
    # Instead of dropping rows, count valid URLs
    valid_urls = df[~df['Gcs Url'].isna()]['Gcs Url'].tolist()
    
    progress_bar = st.progress(0)
    total_videos = len(valid_urls)
    processed_count = 0
    
    if total_videos == 0:
        st.warning("No valid videos to process.")
        return df  # Return the original dataframe without modifications
    
    counter_placeholder = st.empty()
    counter_placeholder.text(f"Processed 0/{total_videos} videos")

    for index, row in df.iterrows():
        video_url = row["Gcs Url"]
        
        # Skip if video_url is None or NaN but keep the row
        if pd.isna(video_url) or not isinstance(video_url, str):
            print(f"Skipping invalid URL at index {index}")
            continue
            
        print(f"Processing video URL: {video_url}")

        # Clear caches periodically (e.g., every 5 videos)
        if processed_count > 0 and processed_count % 5 == 0:
            st.cache_data.clear()
            st.cache_resource.clear()

        video_path = download_and_trim_video(video_url)

        if video_path:
            gif_path = convert_to_gif(video_path, output_dir=output_dir)

            if gif_path:
                resized_gif_path = resize_gif(gif_path)

                if resized_gif_path and os.path.exists(resized_gif_path):
                    upload_gif_to_gcs('tiktok-actor-content', resized_gif_path)
                    os.remove(resized_gif_path)
                    print(f"Deleted GIF: {resized_gif_path}")
                    # Only increment counter after successful processing
                    processed_count += 1
                else:
                    print(f"Resized GIF is still too large or failed to resize: {gif_path}")

            if os.path.exists(video_path):
                os.remove(video_path)
                print(f"Deleted video: {video_path}")

        # Check the GCS URL status
        response = requests.head(video_url)
        if response.status_code != 200:
            print(f"Removing GCS URL and GIF URL for video URL: {video_url}")
            df.at[index, "Gcs Url"] = None
            df.at[index, "Gif Url"] = None

        # Update progress, counter, and collect garbage after each video
        progress = min(processed_count / total_videos, 1.0)  # Ensure progress never exceeds 1.0
        progress_bar.progress(progress)
        counter_placeholder.text(f"Processed {processed_count}/{total_videos} videos")
        gc.collect()

    # Clear caches after completing all processing
    st.cache_data.clear()
    st.cache_resource.clear()
    
    progress_bar.progress(1.0)
    counter_placeholder.text(f"Completed processing all {processed_count}/{total_videos} videos!")

    output_file_name = 'updated_tiktok_urls.xlsx'
    df.to_excel(output_file_name, index=False)
    st.success(f"All videos processed successfully! Processed {processed_count} out of {total_videos} videos.")
    gc.collect()
    return df  # Return the updated dataframe

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
                    yt_shorts_downloader(
                        urls=youtube_shorts,
                        bucket_name="tiktok-actor-content"
                    )
                    gc.collect()
            except Exception as e:
                print(f"An error occurred while downloading YouTube Shorts: {e}")
        
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
                    # For YouTube URLs, create new GCS URL
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
