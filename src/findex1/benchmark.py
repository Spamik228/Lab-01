import argparse
import time
import tracemalloc
from collections import Counter
from pathlib import Path

from findex1.corpus import iter_documents
from findex1.tokenizer import tokenize



def run_eager(data_path: Path, limit: int | None = None) -> dict:

    docs = list(iter_documents(data_path))
    if limit is not None and limit > 0:
        docs = docs[:limit]


    all_tokens = [list(tokenize(doc.text)) for doc in docs]


    doc_count = len(docs)
    total_tokens = sum(len(tokens) for tokens in all_tokens)

    term_counts: Counter[str] = Counter()
    for doc_tokens in all_tokens:
        term_counts.update(doc_tokens)

    vocab_size = len(term_counts)

    return {
        "doc_count": doc_count,
        "total_tokens": total_tokens,
        "vocab_size": vocab_size,
    }



def run_lazy(data_path: Path, limit: int | None = None) -> dict:
    """Обробляє один документ і один токен за раз без збереження в пам'яті."""
    doc_count = 0
    total_tokens = 0
    term_counts: Counter[str] = Counter()

    docs = iter_documents(data_path)
    for i, doc in enumerate(docs):
        if limit is not None and i >= limit:
            break
        doc_count += 1
        for token in tokenize(doc.text):
            total_tokens += 1
            term_counts[token] += 1

    return {
        "doc_count": doc_count,
        "total_tokens": total_tokens,
        "vocab_size": len(term_counts),
    }


def measure(func, data_path: Path, limit: int | None):
    tracemalloc.start()
    start_time = time.perf_counter()

    res = func(data_path, limit)

    elapsed_time = time.perf_counter() - start_time
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return res, elapsed_time, peak_mem / (1024 * 1024)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Eager vs Lazy text processing."
    )
    parser.add_argument(
        "--limit",
        "-N",
        type=int,
        default=1000,
        help="Number of documents to test on.",
    )
    args = parser.parse_args()

    data_path = Path("../../data")

    print(f"Running benchmark on {args.limit} documents...\n")

    _, eager_time, eager_mem = measure(run_eager, data_path, args.limit)
    _, lazy_time, lazy_mem = measure(run_lazy, data_path, args.limit)

    print(
        f"| Version           | Documents | Peak Memory (MB) | Elapsed Time (s) |"
    )
    print(
        f"|-------------------|-----------|------------------|------------------|"
    )
    print(
        f"| eager (lists)     | {args.limit:<9} | {eager_mem:<16.2f} | {eager_time:<16.3f} |"
    )
    print(
        f"| lazy (generators) | {args.limit:<9} | {lazy_mem:<16.2f} | {lazy_time:<16.3f} |"
    )


if __name__ == "__main__":
    main()