import rootutils
import itertools
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import VideoDownloader, FrameCollector

def main():
    input_file = "video_id.txt"
    video_downloader = VideoDownloader()
    frame_collector = FrameCollector()

    video_ids = None
    with open(input_file) as input_file:
        video_ids = input_file.read()
        video_ids = video_ids.split("\n")
        video_ids = [ids.split(" ") for ids in video_ids]
        video_ids = list(itertools.chain(*video_ids))

        input_file.close()

    for video_id in video_ids:
        print("="*70)
        print(" "*20, end=" ")
        print("Downloading : " + video_id)
        print("="*70)
        video_downloader.download_video_yt_dlp(youtube_id=video_id)

    print("="*70)
    print(" "*20, end=" ")
    print("Creating Image")
    for video_id in video_ids:
        print("="*70)
        print(" "*20, end=" ")
        print("Creating image from " + video_id)
        print("="*70)
        frame_collector.get_frames_every_x_frame(video_id=video_id, interval_frames=10)

if __name__ == "__main__":
    main()