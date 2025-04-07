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

# Streamlit UI
st.title("TikTok Video to GIF Converter")

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
country_name = st.text_input("Enter the country name (sheet name)")

if st.button("Process"):
    if uploaded_file and country_name:
        st.write("Reading the Excel file...")
        input_df = pd.read_excel(uploaded_file, sheet_name=country_name)
        input_list_of_dicts = input_df.to_dict(orient="records")

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
            "postURLs": [row["Links"] for row in input_list_of_dicts],
        }

        st.write("Running the Apify actor task...")
        response = asyncio.run(run_actor_task(input_params))
        dataset_id = response.json()["data"]["defaultDatasetId"]
        st.write(f"Actor task completed. Dataset ID: {dataset_id}")

        st.write("Fetching items from the dataset...")
        dataset = asyncio.run(get_items(dataset_id))
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

        output_df = pd.DataFrame(list_of_dicts)
        output_df.to_csv(f"{country_name}_trend_gifs.csv", index=False, encoding="utf_8_sig")
        
        overall_progress.progress(100, text="All videos processed successfully!")
        st.success("Processing complete! Check the generated CSV file for GIF URLs.")

        # Add download button for the CSV file
        with open(f"{country_name}_trend_gifs.csv", "rb") as file:
            st.download_button(
                label="Download CSV File",
                data=file,
                file_name=f"{country_name}_trend_gifs.csv",
                mime="text/csv"
            )

    else:
        st.error("Please upload a file and enter a country name.")