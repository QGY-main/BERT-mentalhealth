# Mental health classification of Reddit posts with BERT

Fine-tunes `bert-base-cased` (max length 512) to classify Reddit posts into 8 classes based on the subreddit they were posted in:
adhd, autism, bipolarreddit, bpd, depression, ptsd, schizophrenia and suicidewatch.

The notebook is `code.ipynb`. It was run on Google Colab with a GPU.

## Data

The data comes from the Reddit Mental Health Dataset (Low et al., 2020). It has posts from 28 subreddits collected between 2018 and 2020, and the label of each post is the subreddit it came from. Only the 8 subreddits above are used in this project.

I used the parquet version hosted on Hugging Face, which has two columns (`label`, `text`) and 1,107,302 rows split across three files:

https://huggingface.co/datasets/kamruzzaman-asif/reddit-mental-health-classification

The original release (CSV files, split by subreddit and time period) is on Zenodo:

https://zenodo.org/records/3941387

### Getting the data

The data files are not included in this repo because they are about 650 MB in total.

The notebook reads the files by name from the folder it runs in, as `train-00000-of-00003`, `train-00001-of-00003` and `train-00002-of-00003`. In Colab you can download them straight into the working folder with:

```
!wget -O train-00000-of-00003 https://huggingface.co/datasets/kamruzzaman-asif/reddit-mental-health-classification/resolve/main/data/train-00000-of-00003.parquet
!wget -O train-00001-of-00003 https://huggingface.co/datasets/kamruzzaman-asif/reddit-mental-health-classification/resolve/main/data/train-00001-of-00003.parquet
!wget -O train-00002-of-00003 https://huggingface.co/datasets/kamruzzaman-asif/reddit-mental-health-classification/resolve/main/data/train-00002-of-00003.parquet
```

## Citation

Low, D. M., Rumker, L., Talkar, T., Torous, J., Cecchi, G., & Ghosh, S. S. (2020). Natural Language Processing Reveals Vulnerable Mental Health Support Groups and Heightened Health Anxiety on Reddit During COVID-19: Observational Study. *Journal of Medical Internet Research, 22*(10), e22635. https://doi.org/10.2196/22635

Dataset: Low, D. M., Rumker, L., Talkar, T., Torous, J., Cecchi, G., & Ghosh, S. S. (2020). Reddit Mental Health Dataset (Version 01) [Data set]. Zenodo. https://doi.org/10.17605/OSF.IO/7PEYQ

## License

The original dataset is released under the Open Data Commons Public Domain Dedication and License (PDDL) v1.0. The posts come from Reddit, so use of the data is also subject to Reddit's terms.

The data contains real posts about mental health, including suicide. It should only be used for research, and the model is not a diagnostic tool.
