import os
import json
from pathlib import Path
from tqdm import tqdm
import joblib
import pandas as pd
import multiprocessing as mp
from multiprocessing import Value, Lock, Manager
from multiprocessing.managers import ListProxy, DictProxy
import numpy as np
from tqdm import tqdm
from typing import Optional
import time
import random
import yt_dlp
import datetime
import platform
import shutil

"""
File defining the YtDlpDownloader, which makes up the bulk of this. It has
features such as error tracking, writing to a status file as the download is in
progress, and being able to skip existing files and/or excluded files in
exclusions.txt
"""

# TQDM bar with number of errors, percentage, how many done

def get_current_time_ms():
    return time.time() * 1000


# https://stackoverflow.com/questions/71326109/how-to-hide-error-message-from-youtube-dl-yt-dlp
class loggerOutputs:
    def error(msg):
        # print("Captured Error: "+msg)
        # lel, we want this thing to SHUT UP
        1
    def warning(msg):
        # print("Captured Warning: "+msg)
        2
    def debug(msg):
        # print("Captured Log: "+msg)
        3

class YtDlpDownloader:
    """
    This class implements the download of the AudioSet dataset.
    It only downloads the audio files according to the provided list of labels and associated timestamps.
    """

    # NOTE: root_path MUST start with ./
    def __init__(
        self,
        labels: list = None,  # None to download all the dataset
        n_jobs: int = 5,
        download_type: str = "unbalanced_train",
        copy_and_replicate: bool = True,
        n_splits: int = 1,
        split_idx: int = 0,
        overwrite_csv: bool = False,
        root_path: str = "./audioset",
        codec_type: str = ".wav",
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

        self.root_path: str = root_path
        self.cached_dir: Path = Path("./cached")

        self.labels: list = labels
        self.n_jobs: int = n_jobs
        self.download_type: str = download_type
        self.copy_and_replicate: bool = copy_and_replicate
        self.n_splits: int = n_splits
        self.split_idx: int = split_idx
        self.codec_type: str = codec_type

        # Get the system information
        system_info = platform.uname()

        # Extract the node name (hostname)
        # for context when looking at the status
        self.hostname = system_info.node

        # sanity checking for number of splits and split index
        assert (
            isinstance(n_splits, int) and n_splits >= 1
        ), "number of splits should be an integer >=1"
        assert (
            isinstance(split_idx, int) and split_idx <= n_splits - 1
        ), "Split index should be in range [0, n_splits)"

        # Create paths
        os.makedirs(self.root_path, exist_ok=True)
        os.makedirs(str(self.cached_dir), exist_ok=True)

        # define URL to get metadata from
        self.segment_csv_url: str = (
            f"http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/{self.download_type}_segments.csv"
        )
        # refers to the "class labels indices" or "labels meta" CSV that one
        # apparently needs to translate the labels from the metadata into human
        # readable labels"
        self.class_label_idxs_url: str = (
            f"http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"
        )

        # local path to save metadata CSV
        self.segment_meta_path: Path = self.cached_dir / Path(
            f"{self.download_type}_segments.csv"
        )
        # local path to save labels meta csv
        self.segment_c2l_path: Path = self.cached_dir / Path("class_labels_indices.csv")
        self.overwrite_csv: bool = overwrite_csv

        # Can change
        self.metadata: pd.DataFrame = None
        self.exclusions: list[str] = []

    def load_segment_csv_url(self):
        """Downloads the segment URL based off the 'download_type' and foregoes the download if the metadata is present in the cache directory"""
        print("Loading metadata CSV...")
        if not self.segment_meta_path.exists():
            # Load the metadata
            print("Cached csv file not detected.")
            print(f"Downloading from {self.segment_csv_url}")
            self.metadata: pd.DataFrame = pd.read_csv(
                self.segment_csv_url,
                sep=",",
                skiprows=3,
                header=None,
                names=["YTID", "start_seconds", "end_seconds", "positive_labels"],
                engine="python",
            )
            print(f"Saved to {self.segment_meta_path}")
            self.metadata.to_csv(str(self.segment_meta_path))

        else:
            self.metadata: pd.DataFrame = pd.read_csv(
                str(self.segment_meta_path),
                sep=",",
                skiprows=3,
                header=None,
                names=["YTID", "start_seconds", "end_seconds", "positive_labels"],
                engine="python",
            )
            print(f"Cached csv file detected at {self.segment_meta_path}")

        print(self.metadata.iloc[[1, 2, 3]])
        self.metadata["positive_labels"] = self.metadata["positive_labels"].apply(
            lambda x: x.replace('"', "")
        )
        self.metadata = self.metadata.reset_index(drop=True)

    def load_class_mapping_csv(self):
        """
        .. because of course they can't have it like a regular (*#$#@! human being


        """
        print("Loading label map CSV...")
        if not self.segment_c2l_path.exists():
            print("Cached label meta csv file not detected.")
            print(f"Downloading from {self.class_label_idxs_url}")
            self.label_meta_df = pd.read_csv(
                self.class_label_idxs_url,
                sep=",",
            )
            self.label_meta_df.to_csv(self.segment_c2l_path)
            print(f"Saved to path {self.segment_c2l_path}")
        else:
            print(
                f"Cached label meta csv file detected at path {self.segment_c2l_path}"
            )
            self.label_meta_df = pd.read_csv(
                self.segment_c2l_path,
                sep=",",
            )

        self.display_to_machine_mapping = dict(
            zip(self.label_meta_df["display_name"], self.label_meta_df["mid"])
        )
        self.machine_to_display_mapping = dict(
            zip(self.label_meta_df["mid"], self.label_meta_df["display_name"])
        )
        return

    def load_exclusions(self, exclusion_path: str = "./exclusions.txt"):
        """Loads exclusions to then use when determining whether to download a clip or not

        Args:
            exclusion_path (str, optional): path to the exclusions file . Defaults to "./exclusions.txt".
            NOTE: the format for the exclusions.txt file should simply be:

            ytid1_startseconds_endseconds.wav
            ytid2_startseconds_endseconds.wav
            ...

        """
        exclusions: list[str] = []

        if not Path(exclusion_path).exists():
            print(f"WARNING, exclusions at {exclusion_path} do not exist!")
        else:

            raw_lines: list[str]
            with open(exclusion_path, "r") as f:
                raw_lines = f.readlines()

            self.exclusions = [x.strip() for x in raw_lines]

        # we're going to trust the user and believe that the ids used in the exclusions are valid

    def percentage_fmt(num: float) -> str:
        return "{:.2%}".format(num)

    def filename_from_id_and_timestmp(
        ytid: str, start_seconds: str, end_seconds: str
    ) -> str:
        # in this case, the start_seconds and end_seconds HAVE to be strings,
        #  or else the naming might get thrown off
        return f"{ytid}_{start_seconds}_{end_seconds}"

    # Value is an integer
    def downloader(
        self,
        id: int,
        chunk: pd.DataFrame,
        total_file_count: DictProxy,
        total_errors: DictProxy,
        lock: Lock,
        update_mod: int = 10,
    ):
        """Function to run in each multiprocess

        Args:
            id (int): id of job
            chunk (pd.DataFrame): subpart of the split of the dataset assigned to the job
            total_file_count (DictProxy): dictionary containing file counts 
            total_errors (DictProxy): dictionary containing errors
            lock (Lock): sempahore used for accessing data
            update_mod (int, optional): How often to update the counts for the UI. Defaults to 10.
        """
        # print("Simulate download")
        local_errors: list[str] = []
        local_files_count: int = 0

        for i, row in chunk.iterrows():
            ytid: str = row["YTID"]
            start_seconds: str = row["start_seconds"]
            end_seconds: str = row["end_seconds"]
            positive_labels: str = row["positive_labels"]
            # print(f"{(ytid, start_seconds, end_seconds, positive_labels)=}")

            filename: str = (
                YtDlpDownloader.filename_from_id_and_timestmp(
                    ytid, start_seconds, end_seconds
                )
                + f".%(ext)s"
            )
            potential_filepaths: list[str] = []

            file_local_exists: bool = False
            file_local_path: str = ""

            clip_labels: list[str] = []
            lcl_fs: list[str] = []
            positive_machine_labels: list[str] = positive_labels.split(",")
                
            path_w_codec : Path = Path(filename).with_suffix(
                    self.codec_type
                )
            
            # saves display labels and checks for all paths the audio should have in theory
            for label in positive_machine_labels:
                display_label = self.machine_to_display_mapping[label]
                clip_labels.append(display_label)
                display_path: Path = self.root_path / Path(display_label)
                os.makedirs(str(display_path), exist_ok=True)
                lcl_path: Path = display_path / path_w_codec
                if lcl_path.exists():
                    lcl_fs.append(lcl_path)

            download : bool = True

            if str(path_w_codec) in self.exclusions:
                # print(f"{filename} excluded, so we won't download it")
                download = False
                # continue
            # if both labels were found to exist locally, skip
            elif len(lcl_fs) == len(positive_machine_labels):
                # print(f"Already exists at {lcl_fs}")
                # print("Skipping...")
                download = False
                # continue

            if download:

                # DON'T need to add the extension since it already has the %(ext)s part at this point
                # fot context, YT-DLP needs the %(ext)s part for some reason
                dwnld_paths: list[Path] = [
                    Path(self.root_path) / Path(lbl) / Path(filename) for lbl in clip_labels
                ]
                # print(f"{dwnld_paths=}")
                result_tuple: tuple[int, Exception] = self.download_audio_section(
                    ytid, start_seconds, end_seconds, dwnld_paths
                )
                result_code, exception = result_tuple
                # print(f"{result_code=}")
                if result_code == 0:
                    local_files_count += 1
                else:
                    local_errors.append(str(exception))
            else:
                local_files_count+=1

            if i % update_mod == 0:
                with lock:
                    # print(f"Local success count {local_files_count}")
                    total_file_count[f"job{id}"] = local_files_count
                    total_errors[f"job{id}"] = local_errors

    def download_audio_section(
        self,
        ytid: str,
        start_time: int,
        end_time: int,
        dwnld_paths: list[Path],
        codec_type: str = "wav",
        quiet: bool = True,
    ) -> tuple[int, Optional[Exception]]:

        url: str = f"https://www.youtube.com/watch?v={ytid}"
        ydl_opts = {
            "quiet": quiet,
            "no_warnings": quiet,  # Suppress warnings if quiet is True
            "format": "bestaudio/best",
            "logger" : loggerOutputs,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": codec_type,
                    "preferredquality": "192",
                }
            ],
            "outtmpl": str(dwnld_paths[0].with_suffix(".%(ext)s")),
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

                if len(dwnld_paths) > 1:
                    for path in dwnld_paths[1:]:
                        shutil.copy(
                            dwnld_paths[0].with_suffix(self.codec_type),
                            path.with_suffix(self.codec_type),
                        )
                return (retcode, None)

            except Exception as e:
                # raise e
                ex = e
                # NOTE: the lack of exception need not imply the video downloaded successfully
                # print("No errors")
                return (1, e)
            else:
                # print("Error")
                return (1, ex)

    def init_multipart_download(
        self, format: str = "wav", quality: int = 5, checkin_debounce: int = 20 * 1000
    ):
        """Initializes the download to be done with a variable amount of workers

        Args:
            format (str, optional): format to download in. Defaults to 'wav'.
            quality (int, optional): Defaults to 5.
            checkin_debounce (int, optional): Time to wait between json status saving. Defaults to 20*1000.
        """

        self.format: str = format
        self.quality: int = quality

        sublen: int = round(len(self.metadata) / self.n_splits)

        # don't care that much about overlap
        bounds: tuple = (sublen * self.split_idx + 1, sublen * (self.split_idx + 1) - 1)

        print(f"n_jobs={self.n_jobs}")
        print(f"n_splits={self.n_splits}")
        print(f"split_idx={self.split_idx}")
        print(f"Total length of dataset is {len(self.metadata)}")
        print(f"Downloading from indices {bounds[0]}  to index {bounds[1]}")
        subset: pd.DataFrame = self.metadata.loc[bounds[0] : bounds[1]]
        print(f"Length of subset to download is {len(subset)}")
        expected_total_files: int = bounds[1] - bounds[0]

        chunks: list[pd.DataFrame] = np.array_split(subset, self.n_jobs)
        print(f"{type(chunks)=}")

        last_time_checkedin: int = get_current_time_ms()

        # for the purpose of being able to use the manager across all of these
        with Manager() as manager:
            job_filescnt_dict: DictProxy = manager.dict()  # list of counts for each job
            job_errors_dict: DictProxy = manager.dict()
            lock: Lock = Lock()

            processes = []
            print("Initializing processes")
            for i in range(self.n_jobs):
                p = mp.Process(
                    target=self.downloader,
                    args=(i, chunks[i], job_filescnt_dict, job_errors_dict, lock),
                )
                processes.append(p)
                p.start()
            # i: int = 0
            # self.downloader(i, chunks[i], job_filescnt_dict, job_errors_dict, lock)

            with tqdm(total=expected_total_files, desc="Total Files") as pbar:
                last_total = 0
                while any(p.is_alive() for p in processes):
                    total_files_cnt: int = 0
                    total_errors_cnt: int = 0

                    for key in job_filescnt_dict.keys():
                        total_files_cnt += job_filescnt_dict[key]
                    for key in job_errors_dict.keys():
                        total_errors_cnt += len(job_errors_dict[key])

                    curr_pcnt: float = float(total_files_cnt) / expected_total_files
                    pbar.update(total_files_cnt - last_total)
                    pbar.set_postfix(
                        errors=total_errors_cnt,
                        percentage=YtDlpDownloader.percentage_fmt(curr_pcnt),
                    )
                    last_total = total_files_cnt

                    # write a checkin
                    if get_current_time_ms() - last_time_checkedin > checkin_debounce:
                        self.save_status_json(
                            job_errors_dict, total_errors_cnt, curr_pcnt
                        )
                        last_time_checkedin: int = get_current_time_ms()

            for p in processes:
                p.join()

            print(f"Total files downloaded: {job_filescnt_dict.value}")
            print(f"Total errors: {len(job_errors_dict)}")

    def save_status_json(self, total_errors, current_errors, curr_pcnt):
        last_status_dict: dict = {
            "Machine Name": self.hostname,
            "download_path": os.path.realpath(self.root_path),
            "n_splits": self.n_splits,
            "split_idx": self.split_idx,
            "current_total": current_errors,
            "errors": [{job: errors} for job, errors in total_errors.items()],
            "num_errors": len(total_errors),
            "current_pcnt": curr_pcnt,
        }

        with open("last_status.json", "w", encoding="utf-8") as f:
            json.dump(last_status_dict, f, indent=4)
