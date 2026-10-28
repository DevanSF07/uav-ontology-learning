# run_mmd_pipeline.py

from config import INPUT_FILES, TOP_K, OUTDIR
from pipeline import process_one_file


def main():
    if not INPUT_FILES:
        print("No INPUT_FILES configured yet. Edit config.py and put your .mmd file paths there.")
        return

    for fp in INPUT_FILES:
        process_one_file(fp, top_k=TOP_K, outdir=OUTDIR)


if __name__ == "__main__":
    main()