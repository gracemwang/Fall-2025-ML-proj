import os
from pathlib import Path
import tempfile
import zipfile

import pandas as pd
import argparse
import re
import unicodedata

import requests


# Translation table: delete ASCII control chars and zero-width chars; map NBSP -> space
_DELETE = {**{i: None for i in range(0x00, 0x20)},  # C0 controls (keep \n,\r,\t via regex reinsert if you like)
           0x7F: None,                              # DEL
           0x200B: None, 0x200C: None, 0x200D: None, 0x2060: None, 0xFEFF: None}  # ZWSP,ZWNJ,ZWJ,WJ,BOM
_NBSP_TO_SPACE = {0x00A0: 0x20}

_WS_COLLAPSE_RE = re.compile(r"\s+")

pd.options.mode.string_storage = 'pyarrow'


min_length_param_name = 'min_length'
drop_short_param_name = 'drop_short'
pickle_param_name = 'pickle'


def download_zip_from_url(url, destination_path):
    """
    Downloads a ZIP file from a given URL and saves it to a specified path.

    Args:
        url (str): The URL of the ZIP file to download.
        destination_path (str): The local path (including filename) where
                                to save the downloaded ZIP file.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; Python requests)",
            "Accept"    : "application/zip, application/octet-stream, */*"
        }
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        with open(destination_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Could not download the ZIP file: {e}")


def generate_dataset_from_remote(language: str, output_file: str | None = None, **kwargs) -> pd.DataFrame:
    """
    Download translation sentence pairs and generate a dataset from them.

    :param language: The language code.
    :param output_file: If given, save the dataset to this file.
    :return: The DataFrame containing the sentence pairs.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            url = f"https://www.manythings.org/anki/{language}-eng.zip"
            filename = Path(tmpdir) / "dataset.zip"
            download_zip_from_url(url, str(filename))
        except Exception as e:
            raise FileNotFoundError(f'Failed to download file for language: {language}\nError{e}')

        with zipfile.ZipFile(str(filename), 'r') as zip_ref:
            zip_ref.extractall(tmpdir)

        try:
            files = [f for f in Path(tmpdir).iterdir() if f.is_file() and f.suffix == '.txt' and f.stem != '_about']
            if len(files) > 1:
                raise RuntimeError(f"Unzipped dataset contains more than the expected number of .txt files: {files}")

            dataset_file = files[0]
            df = stream_to_dataframe(dataset_file, **kwargs)
            df = clean_data(df)
        except:
            raise RuntimeError(f"Failed to parse file for language: {language}")

    if kwargs[drop_short_param_name]:
        df = remove_short_examples(df, **kwargs)

    if output_file:
        save_dataframe(df, output_file, **kwargs)

    return df


def generate_dataset_from_file(input_file: str, output_file: str | None = None, **kwargs) -> pd.DataFrame:
    """
    Read a file of translation sentence pairs and generate a dataset of sentence pairs.

    :return:
    """
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            df = stream_to_dataframe(f, **kwargs)
            df = clean_data(df)
    except FileNotFoundError:
        print(f"File {input_file} not found.")
        raise

    if kwargs[drop_short_param_name]:
        df = remove_short_examples(df, **kwargs)

    if output_file:
        save_dataframe(df, output_file, **kwargs)

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fast, language-agnostic cleaner:
      - Unicode NFKC
      - remove zero-width + control chars
      - NBSP -> space
      - collapse whitespace
    Preserves letters/diacritics in any script.
    """
    def _clean_series(s: pd.Series) -> pd.Series:
        # 1) Normalize (unicodedata.normalize is still Python-level, but we do it once per cell)
        s = s.astype(str).map(lambda x: unicodedata.normalize("NFKC", x))
        # 2) NBSP -> space via translate (C-level)
        s = s.map(lambda x: x.translate(_NBSP_TO_SPACE))
        # 3) Drop controls & zero-width via translate (C-level)
        s = s.map(lambda x: x.translate(_DELETE))
        # 4) Collapse whitespace (vectorized regex)
        s = s.str.replace(_WS_COLLAPSE_RE, " ", regex=True).str.strip()
        return s

    for col in ("English", "Translation"):
        if col in df.columns:
            df[col] = _clean_series(df[col])
    return df


def stream_to_dataframe(stream, drop_attribution: bool = True, **kwargs) -> pd.DataFrame:
    """
    Generate a dataframe from a text stream.

    :param stream: The character stream to read from.
    :param drop_attribution: If True, drop the attribution column.
    :return: The dataframe.
    """
    col_labels = ['English', 'Translation', 'Attribution']
    df = pd.read_csv(stream, sep='\t', header=None, names=col_labels)
    if drop_attribution:
        df = df.drop(columns=['Attribution'])
    return df


def remove_short_examples(dataset: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Remove examples that are shorter than len words.

    :param dataset: The dataset to process.
    :return: The modified DataFrame.
    """
    kwargs[min_length_param_name] = int(kwargs[min_length_param_name])
    if not kwargs[min_length_param_name]:
        raise ValueError(f"Parameter '{min_length_param_name}' is required.")

    minimum = kwargs[min_length_param_name]
    def should_keep(row):
        # return True if the row should be kept
        return len(row['English'].split(' ')) >= minimum and len(row['Translation'].split(' ')) >= minimum

    return dataset[dataset.apply(should_keep, axis=1)]


def save_dataframe(df: pd.DataFrame, output_file: str, **kwargs):
    """
    Save the dataframe to a file.
    :param df: The DataFrame to save.
    :param output_file: The file to save to.
    """
    # Set the filename to 'data/tatoeba/output_file.pkl'
    folder = Path('data') / 'tatoeba'
    folder.mkdir(parents=True, exist_ok=True)

    if kwargs[pickle_param_name]:
        file = folder / (output_file + ".pkl")
        df.to_pickle(file, index=False)
    else:
        file = folder / (output_file + ".csv")
        df.to_csv(file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            prog="PREPROCESS",
            description="Preprocess Tatoeba data into a usable dataset, either from local data or retrieve it from online.")

    parser.add_argument('input', help="Language or file to process.")
    parser.add_argument('-o', '--outfile', help="Output file to save the dataset to.")
    parser.add_argument('-d', '--download', help="Download the dataset from online.", action="store_true")
    parser.add_argument('-m', '--minlength', type=int, help="Minimum length of sentences to include in dataset.", default=5)
    parser.add_argument('--drop-short', help="Drop short sentences before processing.", action='store_true')
    parser.add_argument('-p', '--pickle', help="Save the dataset as a pickle file.", action='store_true')

    args = parser.parse_args()

    kwargs = {
        min_length_param_name: args.minlength,
        drop_short_param_name: args.drop_short,
        pickle_param_name: args.pickle,
    }

    if args.download:
        df = generate_dataset_from_remote(args.input,
                                     args.outfile,
                                     **kwargs)
    else:
        df = generate_dataset_from_file(args.input,
                                        args.outfile,
                                        **kwargs)
