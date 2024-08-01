import os
import json
from pathlib import Path
from tqdm import tqdm
import joblib
import pandas as pd
import multiprocessing as mp
from multiprocessing import Value, Lock, Manager
from multiprocessing.managers import ListProxy
import numpy as np
from tqdm import tqdm
from typing import Optional
import time
import random
import yt_dlp
import datetime
import platform


# TQDM bar with number of errors, percentage, how many done

def get_current_time_ms():
    return time.time() * 1000


class YtDlpDownloader:
    """
    This class implements the download of the AudioSet dataset.
    It only downloads the audio files according to the provided list of labels and associated timestamps.
    """

    def __init__(self, 
                    labels: list = None, # None to download all the dataset
                    n_jobs: int = 5,
                    download_type: str = 'unbalanced_train',
                    copy_and_replicate: bool = True,
                    n_splits : int = 1,
                    split_idx: int = 0,
                    overwrite_csv : bool = False,
                    root_path: str = "audioset",
                    ):
        """
        This method initializes the class.
        :param root_path: root path of the dataset
        :param labels: list of labels to download
        :param n_jobs: number of parallel jobs
        :param download_type: type of download (unbalanced_train, balanced_train, eval)
        :param copy_and_replicate: if True, the audio file is copied and replicated for each label. 
                                    If False, the audio file is stored only once in the folder corresponding to the first label.
        :param n_splits: the number of jobs being executed by other machines to split up the work
        :param split_idx: the index of the split to download. It should be in the range [0, n_splits) (n_splits exclusive)
        """
        # Set the parameters
        print(f"{YtDlpDownloader.__name__} initialized for '{download_type}' split ")
        self.root_path : str = root_path
        self.cached_dir : Path = Path("./cached")

        self.labels : list = labels
        self.n_jobs : int = n_jobs
        self.download_type : str = download_type
        self.copy_and_replicate : bool = copy_and_replicate
        self.n_splits : int = n_splits
        self.split_idx: int = split_idx


        # Get the system information
        system_info = platform.uname()

        # Extract the node name (hostname)
        # for context when looking at the status
        self.hostname = system_info.node

        # sanity checking for number of splits and split index
        assert isinstance(n_splits, int) and n_splits >=1, "number of splits should be an integer >=1"
        assert isinstance(split_idx, int) and split_idx <= n_splits-1, "Split index should be in range [0, n_splits)"

        # Create paths
        os.makedirs(self.root_path, exist_ok=True)
        os.makedirs(str(self.cached_dir), exist_ok=True)

        # define URL to get metadata from
        self.segment_csv_url : str = f"http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/{self.download_type}_segments.csv"
        # refers to the "class labels indices" or "labels meta" CSV that one
        # apparently needs to translate the labels from the metadata into human
        # readable labels"
        self.class_label_idxs_url : str = f"http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"

        # local path to save metadata CSV
        self.segment_meta_path : Path = self.cached_dir / Path(f"{self.download_type}_segments.csv")
        # local path to save labels meta csv
        self.segment_c2l_path : Path = self.cached_dir / Path("class_labels_indices.csv")
        self.overwrite_csv : bool = overwrite_csv

        # Can change
        self.metadata : pd.DataFrame = None
        self.exclusions : list[str] = None

    
    def load_segment_csv_url(self):
        """Downloads the segment URL based off the 'download_type' and foregoes the download if the metadata is present in the cache directory
        """
        if not self.segment_meta_path.exists():
            # Load the metadata
            print("Cached csv file not detected.")
            print(f"Downloading from {self.segment_csv_url}")
            self.metadata : pd.DataFrame = pd.read_csv(
                self.segment_csv_url,
                sep=',', 
                skiprows=3,
                header=None,
                names=['YTID', 'start_seconds', 'end_seconds', 'positive_labels'],
                engine='python'
            )
            print(f"Saved to {self.segment_meta_path}")
            self.metadata.to_csv(str(self.segment_meta_path))

        else:
            self.metadata : pd.DataFrame = pd.read_csv(
                str(self.segment_meta_path),
                sep=',', 
                skiprows=3,
                header=None,
                names=['YTID', 'start_seconds', 'end_seconds', 'positive_labels'],
                engine='python'
            )
            print(f"Cached csv file detected at {self.segment_meta_path}")

        print(self.metadata.iloc[[1,2,3]])
        self.metadata['positive_labels'] = self.metadata['positive_labels'].apply(lambda x: x.replace('"', ''))
        self.metadata = self.metadata.reset_index(drop=True)
        
    def load_class_mapping_csv(self):
        """
        .. because of course they can't have it like a regular (*#$#@! human being

        
        """
        if not self.segment_c2l_path.exists():
            print("Cached label meta csv file not detected.")
            print(f"Downloading from {self.class_label_idxs_url}")
            self.label_meta_df = pd.read_csv(
                self.class_label_idxs_url, 
                sep=',',
            )
            self.label_meta_df.to_csv(self.segment_c2l_path)
            print(f"Saved to path {self.segment_c2l_path}")
        else:
            print(f"Cached label meta csv file detected at path {self.segment_c2l_path}")
            self.label_meta_df = pd.read_csv(
                self.segment_c2l_path,
                sep=',',
            )

        self.display_to_machine_mapping = dict(zip(self.label_meta_df['display_name'], self.label_meta_df['mid']))
        self.machine_to_display_mapping = dict(zip(self.label_meta_df['mid'], self.label_meta_df['display_name']))
        return
        
    def load_exclusions(self, exclusion_path : str = "./exclusions.txt"):
        """Loads exclusions to then use when determining whether to download a clip or not

        Args:
            exclusion_path (str, optional): path to the exclusions file . Defaults to "./exclusions.txt".
            NOTE: the format for the exclusions.txt file should simply be:

            ytid1_startseconds_endseconds.wav
            ytid2_startseconds_endseconds.wav
            ...

        """
        exclusions : list[str] = []
        
        if not Path(exclusion_path).exists():
            print(f"WARNING, exclusions at {exclusion_path} do not exist!")
        else:

            raw_lines : list[str]
            with open(exclusion_path, "r") as f:
                raw_lines = f.readlines()
            
            exclusions = [x.strip().replace("\n", "") for x in raw_lines]
        
        # we're going to trust the user and believe that the ids used in the exclusions are valid
        print(f"{exclusions=}")
    def percentage_fmt(num : float) -> str:
        return "{:.2%}".format(num)

    # Value is an integer
    def downloader(self, id : int ,chunk : pd.DataFrame, total_files : Value, total_errors : ListProxy, total_skips : ListProxy, lock : Lock, update_mod : int = 10):
        print(f"{type(chunk)=}")
        # print("Simulate download")
        local_files = 0
        local_errors = []
        
        for i, row in chunk.iterrows():
            time.sleep(random.uniform(0.1, 0.5))  # Simulate download time
            ytid : str = row['YTID']
            start_seconds : int = row['start_seconds']
            end_seconds : int = row['end_seconds']
            positive_labels : str = row["positive_labels"]
            filename : str = f"{os.path.join}"
            print(f"{(ytid, start_seconds, end_seconds, positive_labels)=}")

            for label in positive_labels.split(',')[1:]:
                display_label = self.machine_to_display_mapping[label]

            # if(self.exclusions
            
            if random.random() < 0.9:  # 90% success rate
                local_files += 1
            else:
                local_errors.append([f"Error{id}"])
            if i % update_mod == 0:
            
                with lock:
                    total_files.value += local_files
                    total_errors.extend(local_errors)

    def download_audio_section(
        ytid: str,
        start_time: int,
        end_time: int,
        output_filepath: str,
        codec_type: str = "wav",
        quiet: bool = True,
    ) -> tuple[int, Optional[Exception]]:

        url: str = f"https://www.youtube.com/watch?v={ytid}"
        ydl_opts = {
            "quiet": quiet,
            "no_warnings": quiet,  # Suppress warnings if quiet is True
            "format": "bestaudio/best",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": codec_type,
                    "preferredquality": "192",
                }
            ],
            "outtmpl": output_filepath,
            "download_ranges": lambda info, _: [
                {
                    "start_time": start_time,
                    "end_time": end_time,
                }
            ],
        }

        ex: Exception = None

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                retcode: int = ydl.download([url])
            except Exception as e:
                ex = e

            if ex is None:
                # NOTE: the lack of exception need not imply the video downloaded successfully
                # print("No errors")
                return (retcode, None)
            else:
                # print("Error")
                return (1, ex)
        
    def init_download(
        self,
        format: str = 'wav',
        quality: int = 5,    
        checkin_debounce : int = 20 * 1000 
    ):
        """
        This method downloads the dataset using the provided parameters.
        :param format: format of the audio file (vorbis, mp3, m4a, wav), default is vorbis
        :param quality: quality of the audio file (0: best, 10: worst), default is 5
        """

        self.format : str = format
        self.quality : int = quality

        sublen : int = round(len(self.metadata)/self.n_splits)

        # don't care that much about overlap
        bounds : tuple = (sublen * self.split_idx + 1 , sublen * (self.split_idx + 1)-1)
        
        print(f"Total length of dataset is {len(self.metadata)}")
        print(f"Downloading from indices {bounds[0]}  to index {bounds[1]}")
        subset : pd.DataFrame = self.metadata.loc[bounds[0]: bounds[1]]
        print(f"Length of subset to download is {len(subset)}")
        expected_total_files : int = bounds[1] - bounds[0]

        chunks : list[pd.DataFrame]  = np.array_split(subset, self.n_jobs)
        print(f"{type(chunks)=}")


        last_time_checkedin : int = get_current_time_ms()

        # for the purpose of being able to use the manager across all of these 
        with Manager() as manager:
            total_files : Value = Value('i', 0)
            total_errors : ListProxy = manager.list()
            total_skips : ListProxy = manager.list()
            lock : Lock = Lock()

            processes = []
            print("Initializing processes")
            for i in range(self.n_jobs):
                p = mp.Process(target=self.downloader, args=(i, chunks[i], total_files, total_errors,total_skips, lock))
                processes.append(p)
                p.start()

            with tqdm(total=expected_total_files, desc="Total Files") as pbar:
                last_total = 0
                while any(p.is_alive() for p in processes):
                    current_total : int  = total_files.value
                    current_errors : int = len(total_errors)
                    curr_pcnt : float = (float(current_total)/expected_total_files)
                    pbar.update(current_total - last_total)
                    pbar.set_postfix(errors=current_errors, percentage=YtDlpDownloader.percentage_fmt(curr_pcnt))
                    last_total = current_total
                    time.sleep(0.1)
                    
                    # write a checkin
                    if(get_current_time_ms() - last_time_checkedin > checkin_debounce):
                        self.save_status_json(total_errors, current_errors, curr_pcnt)
                        last_time_checkedin : int = get_current_time_ms()

            for p in processes:
                p.join()

            print(f"Total files downloaded: {total_files.value}")
            print(f"Total errors: {len(total_errors)}")

    def save_status_json(self, total_errors, current_errors, curr_pcnt):
        last_status_dict : dict = {
                                "Machine Name" : self.hostname,
                                "download_path" : os.path.realpath(self.root_path),
                                "n_splits" : self.n_splits,
                                "split_idx" : self.split_idx,
                                "current_total" : current_errors,
                                "errors" : [str(error) for error in total_errors],
                                "num_errors" : len(total_errors),
                                "current_pcnt" : curr_pcnt
                            }

        with open("last_status.json", "w", encoding="utf-8") as f:
            json.dump(last_status_dict, f, indent=4)