import json
import ast
from typing import Any
import pandas as pd
import argparse
import numpy as np
import pickle

from pathlib import Path
from comet import download_model, load_from_checkpoint
from comet.models.utils import Prediction
from torch import set_float32_matmul_precision

project_path = Path('/home/export/doriancl/code/Fall-2025-ML-proj')

set_float32_matmul_precision('medium')


def safe_parse_json(json_string: str) -> dict:
    """
    Parse JSON-like strings into a Python dictionary.

    This is safe to malformed JSON.

    Args:
        json_string: The string to parse.

    Returns:
    A Python dictionary with the JSON's contents, if parseable. Otherwise, empty dict.
    """
    try:
        return json.loads(json_string)
    except Exception:
        pass

    try:
        return ast.literal_eval(json_string)
    except Exception:
        pass

    return {}


PAIRS = [
    ('"', '"'),
    ("'", "'"),
    ('„', '“'),
    ('„', '”'),
    ('‚', '‘'),
    ('‚', '’'),
    (',', "'"),
    (',', '’'),
    ('«', '»'),
    ('‹', '›'),
    ('<', '>'),
    ('{', '}'),
    ('[', ']'),
    ('(', ')'),
]


def strip_all_wrapping(s):
    s = str(s).strip()
    while True:
        if not s:
            return s
        changed = False
        for open_q, close_q in PAIRS:
            if s.startswith(open_q) and s.endswith(close_q):
                s = s[len(open_q):-len(close_q)].strip()
                changed = True
                break
        if not changed:
            return s


def json_pyify(x):
    if isinstance(x, (np.integer, np.int64)):
        return int(x)
    if isinstance(x, (np.floating, np.float64)):
        return float(x)
    if isinstance(x, (np.ndarray,)):
        return x.tolist()
    return x


def preprocess_monolith_output(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Parses the JSON output by the model for the 'monolithic' prompt type, separating translations and confidences into two columns.
    Args:
        df: The raw dataframe of all the model's predictions.
        **kwargs: Other kwargs.

    Returns:
    A processed DataFrame.
    """
    df['AI_Translation'] = df['AI_Translation'].apply(safe_parse_json)
    df = pd.concat([df.drop(columns=["AI_Translation"]),
                    pd.json_normalize(df["AI_Translation"])],
                   axis=1)
    df = df.rename(columns={"translation": "AI_Translation", "confidence": "Confidence"})
    print(df.columns)
    return df


def preprocess(df: pd.DataFrame, **kwargs) -> tuple[pd.DataFrame, dict]:
    """
    Clean up the data so evaluations work as expected.
    
    - Filter confidence for NaN
    - Allow for text wrapped in "".."", "..", '..', ..
    - Malformatted JSON
    - Ignore [ERROR]

    Args:
        df: The data to be cleaned.
        **kwargs: CLI arguments.

    Returns:
    A tuple of the cleaned dataframe and information about the cleanup.
    """
    raw_size = df.shape[0]

    # Drop NaN that may have been created during JSON parsing.
    malformatted_json = df[['AI_Translation', 'Confidence']].isna().any(axis=1).sum()
    print(f"malformatted json: {malformatted_json}")
    df = df.dropna(subset=['AI_Translation', 'Confidence'])

    # Drop other NaN, if they exist for some reason
    other_nan = df.isna().any(axis=1).sum()
    df = df.dropna()

    # Coerce confiences to numeric. This will make NaN, which we drop.
    df['Confidence'] = pd.to_numeric(df['Confidence'], errors='coerce')
    nan_confidences = df.isna().any(axis=1).sum()
    df = df.dropna()

    # Drop translations where the output was [ERROR]
    df = df[df['AI_Translation'] != "[ERROR]"]
    errors = raw_size - df.shape[0]

    # Strip quotation marks from text.
    df['AI_Translation'] = df['AI_Translation'].apply(strip_all_wrapping)
    df['English'] = df['English'].apply(strip_all_wrapping)
    df['Original_Translation'] = df['Original_Translation'].apply(strip_all_wrapping)

    if kwargs['n_samples']:
        df = df.sample(kwargs['n_samples'])

    return df, {
        'errors': errors,
        'malformatted_json': malformatted_json,
        'nan_confidences': nan_confidences,
        'other_nan': other_nan,
    }


def get_scores(df: pd.DataFrame, **kwargs) -> tuple[pd.DataFrame, Prediction]:
    """
    Get 'ground truth' scores for each translation in an experiment output dataframe.

    Args:
        df: The experimental output.
        **kwargs: The CLI arguments.

    Returns:
    A tuple of (the DataFrame with scores column added) and (the whole model output)
    """
    # Use UniTE model, as it appears to have better accuracy
    # https://aclanthology.org/2022.acl-long.558/
    eval_model_path = download_model('Unbabel/wmt22-unite-da')
    eval_model = load_from_checkpoint(eval_model_path)

    data = df.rename(
        columns={
            "English": "src",
            "AI_Translation": "mt",
            "Original_Translation": "ref"
        }
    ).to_dict(orient="records")

    model_output = eval_model.predict(data, batch_size=kwargs['max_concurrent'])
    df['Score'] = model_output.scores

    return df, model_output


def get_filepaths(kwargs: dict[str, Any]) -> tuple[Path, str, Path, Path, str, str]:
    """
    Set up for evaluations.
    Args:
        kwargs: CLI arguments

    Returns:
    Tuple of directories and configuration.
    """
    translation_model = kwargs['model']
    language = kwargs['language']
    thinking = 'CoT' if kwargs['thinking'] else 'Basic'
    prompt_type = kwargs['prompt_type']

    output_dir = project_path / 'data/tatoeba/output'
    eval_dir = project_path / 'data/tatoeba/eval'
    model_subdir = Path(translation_model.lower()) / thinking / prompt_type
    return eval_dir, language, model_subdir, output_dir, translation_model, thinking


def get_translations_df(language: str, output_dir: Path, model_subdir: Path) -> pd.DataFrame:
    """
    Read the translations for a given language to a DataFrame.

    Args:
        output_dir: The directory of all experiment outputs.
        model_subdir: The branch for the given model.
        language: The language.

    Returns:
    A DataFrame of the experiment output.
    """
    translations_dir = output_dir / model_subdir
    translations = [p for p in translations_dir.iterdir() if p.suffix == '.csv']
    output_path = ""

    for translation in translations:
        if language and language not in translation.name:
            continue

        output_path = translation

    df = pd.read_csv(output_path)
    return df


def save_evals(df: pd.DataFrame,
               preprocess_info: dict,
               eval_output: Prediction,
               eval_dir: Path, 
               thinking: str,
               translation_model: str,
               language: str,
               prompt_type: str) -> None:
    """
    Save the evaluation.
    """
    (eval_dir / translation_model).parent.mkdir(parents=True, exist_ok=True)
    eval_file = "_".join([
        translation_model,
        thinking,
        prompt_type,
        language
    ])

    eval_csv = eval_dir / f"{eval_file}.csv"
    eval_info = eval_dir / f"{eval_file}.json"
    eval_pkl = eval_dir / f"{eval_file}.pkl"

    df.to_csv(eval_csv, index=False)
    with open(eval_info, "w") as f:
        preprocess_info = json.loads(json.dumps(preprocess_info, default=json_pyify))
        json.dump(preprocess_info, f, indent=4, sort_keys=True)
    with open(eval_pkl, "wb") as f:
        pickle.dump(eval_output, f)


def evaluate_translations(**kwargs):
    eval_dir, language, model_subdir, output_dir, translation_model, thinking = get_filepaths(kwargs)

    df = get_translations_df(language=language, output_dir=output_dir, model_subdir=model_subdir)

    if kwargs['prompt_type'] == 'monolithic':
        df = preprocess_monolith_output(df, **kwargs)
        print(df.columns)

    df, info = preprocess(df, **kwargs)

    df, eval_output = get_scores(df, **kwargs)

    save_evals(df=df,
               preprocess_info=info,
               eval_output=eval_output,
               eval_dir=eval_dir,
               thinking=thinking,
               translation_model=translation_model,
               prompt_type=kwargs['prompt_type'],
               language=language)

    print(f"Successfully finished scoring translations for {translation_model}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', type=str)
    parser.add_argument('--model', type=str, default='Qwen/Qwen3-1.7B')
    parser.add_argument('--max_concurrent', '-C', type=int, default=512)
    parser.add_argument('--prompt-type', '-T', type=str, choices=['iterative', 'monolithic'], default='iterative')
    parser.add_argument('--thinking', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--n_samples', type=int)

    args = parser.parse_args()

    evaluate_translations(**vars(args))