import yt_dlp
import os
import cv2

from pytube import YouTube

from src.constants import (
    MP4_SUFFIX,
    MP4_INPUT_DIR, 
    YOUTUBE_DIR,
    DATA_DIR,
    IMAGE_DIR
)

class VideoDownloader:
    def __init__(self):
        try:
            # Create directories if they don't exist
            os.makedirs(MP4_INPUT_DIR)
        except OSError as e:
            if os.path.isdir(MP4_INPUT_DIR):  # Handles existing folder case
                print(f"Folder '{MP4_INPUT_DIR}' already exists.")
            else:
                print(f"Error creating folder: {e}")

    def download_video_yt_dlp(self, youtube_id):
        """
        Downloads a video from YouTube by its unique identifier using yt_plp library and save it to your device
        Args:
            youtube_id (str): Identifier of video on YouTube
        """
        if not os.path.exists(MP4_INPUT_DIR):
            os.makedirs(MP4_INPUT_DIR)
        output_path = os.path.join(MP4_INPUT_DIR, youtube_id + MP4_SUFFIX)

        # Ensure that we only download video at most once.
        if os.path.exists(output_path):
            return

        video_url = YOUTUBE_DIR + youtube_id
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best',  # Fallback chain
            'outtmpl': output_path,
            'quiet': False,  # Set to False to see warnings/errors for debugging
            'merge_output_format': 'mp4',  # Ensures merged output is MP4
            'noplaylist': True,  # Prevents downloading playlists if URL is misinterpreted
            'retries': 10,  # Retry on transient errors
            'extractor_retries': 10,  # Retry extraction on failure
            'http_headers': {  # Mimic a real browser to avoid restrictions
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            },
        }

        print("Url: ", video_url)
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                ydl.download([video_url])
                print('Video was downloaded successfully using yt_plp library')
            except yt_dlp.DownloadError as e:
                print('Error downloading video:', e)

    def download_video_pytube(self, youtube_id):
        """
        Downloads a video from YouTube by its unique identifier using pytube library and save it to your device
        Args:
            youtube_id (str): Identifier of video on YouTube
        """
        youtube_url = YOUTUBE_DIR + youtube_id
        try:
            yt = YouTube(youtube_url)
        except:
            print("Connection error")

        mp4_streams = yt.streams.filter(file_extension=MP4_SUFFIX).all()

        d_video = mp4_streams[-1]
        if not os.path.exists(MP4_INPUT_DIR):
            os.makedirs(MP4_INPUT_DIR)
        output_path = os.path.join(MP4_INPUT_DIR, youtube_id + MP4_SUFFIX)
        try:
            d_video.download(output_path=output_path)
            print('Video was downloaded successfully using pytube library')

        except:
            print("Error downloading video")

class FrameCollector:
    def __init__(self):
        pass
    
    def get_frames_every_x_frame(self, video_id, interval_frames=2, output_dir=os.path.join(DATA_DIR, IMAGE_DIR)):
        """
        Extracts frames from a video at a specified interval.

        Args:
            video_id (str): The id of video.
            interval_seconds (int, optional): Interval in seconds between frames. Defaults to 2.
            output_dir (str, optional): Path to the directory for saving the JPEG images. Defaults to FRAME_DIR in RESULT_DIR.
            save (bool, optional): Save frames form extraction or no. Defaults to True.

        Returns:
            bool: True if successful, False otherwise.
        """
        if interval_frames <= 0:
            print("Error: Interval must be a positive number.")
            return False

        video_path = os.path.join(MP4_INPUT_DIR, video_id + MP4_SUFFIX)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        if not cap.isOpened():
            print("Error opening video file")
            return False

        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indexes = range(0, num_frames, interval_frames)

        return self.__get_frames_at_indexes_opencv(video_path, video_id, frame_indexes, output_dir=output_dir)

    def __get_frames_at_indexes_opencv(self, video_path, video_id, frame_indexes, output_dir):
        """
        Extracts frames from a video at specific durations and saves them to a directory.

        Args:
            video_path: The path to the video file.
            video_id: The id of video.
            output_dir: The directory to save the frames.
            indexes: A list of indexes at which to extract frames.

        """

        self.__create_output_dir(output_dir)
        
        cap = cv2.VideoCapture(video_path)

        # Check if the video capture was successful
        if not cap.isOpened():
            print("Error opening video file")
            return False

        # Get video FPS
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Extract frames at specified timestamps
        for frame_index in frame_indexes:

            # Set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))

            ret, frame = cap.read()
            if ret:
                output_path = f"{output_dir}/{video_id}_{frame_index:.2f}.jpg"
                cv2.imwrite(output_path, frame)
            else:
                print(f"Error extracting frame at {frame_index}")

        cap.release()
    
    def __create_output_dir(self, output_dir):
        try:
            os.makedirs(output_dir)
        except OSError as e:
            if os.path.isdir(output_dir):
                print(f"Folder '{output_dir}' already exists.")
            else:
                print(f"Error creating folder: {e}")