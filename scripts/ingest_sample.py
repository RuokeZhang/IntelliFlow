import argparse
from pathlib import Path

from app.rag.ingest import ingest_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "sample_data" / "sample.txt"),
        help="要入库的文件路径",
    )
    args = parser.parse_args()
    doc_id, count = ingest_file(args.path, source="sample")
    print(f"ingested doc={doc_id}, chunks={count}")


if __name__ == "__main__":
    main()

