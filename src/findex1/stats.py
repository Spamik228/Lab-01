import argparse
import itertools
import logging
import time
import tracemalloc
from collections import Counter
from pathlib import Path

from findex1.corpus import iter_documents
from findex1.tokenizer import tokenize

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def compute_stats(data_path: Path, limit: int | None = None) -> dict:

    doc_count = 0
    total_tokens = 0
    term_counts: Counter[str] = Counter()

    documents = iter_documents(data_path)


    if limit is not None and limit > 0:
        documents = itertools.islice(documents, limit)

    for doc in documents:
        doc_count += 1
        for token in tokenize(doc.text):
            total_tokens += 1
            term_counts[token] += 1

    vocab_size = len(term_counts)
    top_50 = term_counts.most_common(50)

    return {
        "doc_count": doc_count,
        "total_tokens": total_tokens,
        "vocab_size": vocab_size,
        "top_50": top_50,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Stream statistics computation for text corpus."
    )
    parser.add_argument(
        "data_path",
        type=Path,
        nargs="?",
        default=Path("data"),
        help="Path to corpus directory (default: data/)",
    )
    parser.add_argument(
        "--limit",
        "-N",
        type=int,
        default=None,
        help="Limit execution to the first N documents for testing/development.",
    )

    args = parser.parse_args()

    if not args.data_path.exists():
        logger.error(f"Directory '{args.data_path}' does not exist.")
        return


    tracemalloc.start()
    start_time = time.perf_counter()

    stats = compute_stats(args.data_path, limit=args.limit)

    elapsed_time = time.perf_counter() - start_time
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()


    print("\n" + "=" * 50)
    print("           CORPUS STREAM STATISTICS           ")
    print("=" * 50)
    print(f"Documents processed : {stats['doc_count']:,}")
    print(f"Total tokens        : {stats['total_tokens']:,}")
    print(f"Vocabulary size     : {stats['vocab_size']:,}")
    print(f"Elapsed time        : {elapsed_time:.3f} seconds")
    print(f"Peak memory usage   : {peak_mem / (1024 * 1024):.2f} MB")
    print("-" * 50)
    print("Top-50 Most Frequent Terms:")
    print("-" * 50)

    for rank, (term, count) in enumerate(stats["top_50"], 1):
        print(f"{rank:2d}. {term:<20} {count:,}")

    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()