"""
download_cifake.py
"""

import kagglehub

DATASET = "birdy654/cifake-real-and-ai-generated-synthetic-images"


def main() -> str:
    path = kagglehub.dataset_download(DATASET)
    print("Path to dataset files:", path)
    return path


if __name__ == "__main__":
    main()
