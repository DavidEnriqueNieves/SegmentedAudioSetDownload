from downloader import YtDlpDownloader
from argparse import ArgumentParser, Namespace
import yt_dlp
from yt_dlp.utils import download_range_func, DownloadError
from typing import Optional
import yt_dlp
import time

"""
A script made because I can't trust Open Source Code and I'm competent
Based off 
"""

if __name__ == "__main__":

    argparser: ArgumentParser = ArgumentParser()
    argparser.add_argument("--n_splits", type=int)
    argparser.add_argument("--split_idx", type=int)
    argparser.add_argument("--n_jobs", type=int)
    argparser.add_argument("--debug", action="store_true")
    args: Namespace = argparser.parse_args()

    if args.debug:
        import debugpy
        PORT : int = 5678
        debugpy.listen(5678)
        print(f"Waiting for debug client on port {PORT}")
        debugpy.wait_for_client()

    n_splits: int = 1
    split_idx: int = 0
    n_jobs : int = 5

    if args.n_splits:
        n_splits: int = args.n_splits

    if args.split_idx:
        split_idx: int = args.split_idx
    
    if args.n_jobs:
        n_jobs : int = args.n_jobs

    print("Downloading the 'unbalanced_train' split in the wav file format...")
    print(f"{split_idx=}")
    print(f"{n_splits=}")

    # NOTE: the root path MUST have a forward slash
    d : YtDlpDownloader = YtDlpDownloader(root_path='./audioset/', n_jobs=n_jobs, download_type='unbalanced_train', copy_and_replicate=False, n_splits=n_splits, split_idx=split_idx, overwrite_csv=False)
    d.load_segment_csv_url()
    d.load_class_mapping_csv()
    d.load_exclusions()
    d.init_multipart_download()

    # # https://stackoverflow.com/questions/73516823/using-yt-dlp-in-a-python-script-how-do-i-download-a-specific-section-of-a-video
    # # Example usage
    # start_time = 29  # 1 minute 30 seconds
    # end_time = 49  # 3 minutes 45 seconds
    # output_filename = "output.%(ext)s"

    # ytid : str = "kGU_-fnSQI8"
    # download_audio_section(ytid, start_time, end_time, output_filename)
