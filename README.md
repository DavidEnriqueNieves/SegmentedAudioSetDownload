## AudioSetDownloader

Based off MorenoLaQuatra's [original repo](https://github.com/MorenoLaQuatra/audioset-download).


Run with `python3 download_unbalanced_train.py --split_idx 0 --n_splits 4 --n_jobs 12`, and feel free to tinker.


NOTE: `--sleep_amount` is in seconds, and by default, the errors that are NOT caused by bot sniping will be saved to `unavailable_urls.txt` by default. When running the download again, it will use that file to skip over the download phase of that to avoid incurring any unnecessary downloads.