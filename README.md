# Fall-2025-ML-proj


## Getting Started

1. Conda is outdated. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Update the uv project: `$ uv sync`

## Preprocessing Tatoeba data

For data pulled from [ManyThings.org][many-things], data can be preprocessed using `preprocess_tatoeba.py`. These data are a selection of translation sentence pairs from the [Tatoeba Project](http://tatoeba.org/home).

This script allows either parsing a local file downloaded from [ManyThings.org][many-things], or to download it from the website before parsing.

[many-things]: https://www.manythings.org/anki/

### CLI Usage

The only required argument is the name of the file to download or parse. However, this is much more functional by using the optional arguments, e.g.

```shell
python preprocess_tatoeba.py -o LANG_PREFIX --drop-short --download LANG_PREFIX
```

To see all arguments, pass `--help`:

```shell
python preprocess_tatoeba.py --help
```

### Module usage

The functions can also be called directly, e.g. to get a dataframe of French-English translations, we can do:

```python
from preprocess_tatoeba import generate_dataset_from_remote

kwargs = {
    'min_length': 5,
    'drop_short': True,
    'pickle': True
}

df = generate_dataset_from_remote('fra', output_file=None, **kwargs)
```