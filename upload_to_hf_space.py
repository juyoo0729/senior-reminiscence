# -*- coding: utf-8 -*-
"""Upload this Streamlit app to the existing Hugging Face Space.

Before running:
1. Install the client: pip install huggingface_hub
2. Login once: huggingface-cli login
3. Set OPENAI_API_KEY as a Space secret in the Hugging Face UI.

Only the runtime data files needed by the app are uploaded from data/.
The local .env file is always excluded.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import upload_folder


DEFAULT_REPO_ID = "midi3008/senior-reminiscence"

ALLOW_PATTERNS = [
    "README.md",
    "requirements.txt",
    "app_companion.py",
    "storybook.py",
    ".env.example",
    ".streamlit/config.toml",
    "src/**",
    "images/**",
    "data/corpus.faiss",
    "data/corpus_meta.pkl",
    "data/clf_5cat.joblib",
    "data/embed_model.txt",
]

IGNORE_PATTERNS = [
    ".git/**",
    ".env",
    "__pycache__/**",
    "**/__pycache__/**",
    "*.pyc",
    ".venv/**",
    "venv/**",
    "data/train_5cat.csv",
    "data/valid_5cat.csv",
    "data/corpus_50k.faiss",
    "data/corpus_meta_50k.pkl",
    "data/kobert_5cat/**",
    '"; print(pandas.__version__)•"',
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload the companion app to Hugging Face Spaces."
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument(
        "--message",
        default="Update Streamlit companion app",
        help="Commit message shown on Hugging Face Hub.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    upload_folder(
        repo_id=args.repo_id,
        repo_type="space",
        folder_path=str(root),
        path_in_repo=".",
        allow_patterns=ALLOW_PATTERNS,
        ignore_patterns=IGNORE_PATTERNS,
        commit_message=args.message,
    )


if __name__ == "__main__":
    main()
