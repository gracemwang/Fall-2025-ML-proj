import pandas as pd
from comet import download_model, load_from_checkpoint
from pathlib import Path
import argparse
import pickle


project_path = Path('/home/export/doriancl/code/Fall-2025-ML-proj')


def evaluate_qwen2_5_0_5b_instruct(model, translation_file_path: Path):
    df = pd.read_csv(translation_file_path)

    data = df.rename(
        columns={
            "English": "src",
            "AI_Translation": "mt",
            "Original_Translation": "ref"
        }
    ).to_dict(orient="records")

    model_output = model.predict(data, batch_size=256)
    return model_output.scores

def evaluate_translations(translation_model: str,
                          language: str | None):
    # Use UniTE model, as it appears to have better accuracy
    # https://aclanthology.org/2022.acl-long.558/
    eval_model_path = download_model('Unbabel/wmt22-unite-da')
    eval_model = load_from_checkpoint(eval_model_path)

    translations_dir = project_path / 'data/tatoeba/output' / translation_model
    translations = [p for p in translations_dir.iterdir() if p.suffix == '.csv']

    evals = {}

    for translation in translations:
        if language and language not in translation.name:
            continue

        if translation_model.lower() == 'qwen/qwen2.5-0.5b-instruct':
            evals[translation] = evaluate_qwen2_5_0_5b_instruct(eval_model, translation)

    pklname = f'{language}_scores.pkl' if language else 'translation_scores.pkl'
    scores_file = translations_dir / pklname
    with open(scores_file, "wb") as f:
        pickle.dump(evals, f)

    print(f"Successfully finished scoring translations for {translation_model}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', type=str)
    parser.add_argument('--model', type=str, default='qwen/qwen2.5-0.5b-instruct')

    args = parser.parse_args()

    evaluate_translations(translation_model=args.model,
                          language=args.language)