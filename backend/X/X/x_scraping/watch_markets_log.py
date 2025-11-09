# watch_markets_log.py

import argparse
import os
import time
from pathlib import Path


def tail_lines(path: Path, n: int) -> list[str]:
    """
    Return the last n lines of the file at path.
    Works reasonably for typical log sizes.
    """
    if not path.exists():
        return []

    # simple implementation: read whole file once
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    if n <= 0 or n >= len(lines):
        return [line.rstrip("\n") for line in lines]
    return [line.rstrip("\n") for line in lines[-n:]]


def follow(path: Path, start_at_end: bool = True):
    """
    Generator that yields new lines as they are written.
    Similar to `tail -f`.
    """
    if not path.exists():
        # wait until file appears
        print(f"[watch] waiting for {path} to be created...")
        while not path.exists():
            time.sleep(1.0)

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        if start_at_end:
            f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.5)
                continue
            yield line.rstrip("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Watch markets.log for new markets and resolutions"
    )
    parser.add_argument(
        "-f",
        "--follow",
        action="store_true",
        help="keep watching for new lines (like tail -f)",
    )
    parser.add_argument(
        "-n",
        "--lines",
        type=int,
        default=40,
        help="number of lines to show from the end (default 40)",
    )
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        default="markets.log",
        help="path to log file (default markets.log)",
    )
    args = parser.parse_args()

    log_path = Path(args.path).resolve()

    if not log_path.exists():
        print(f"{log_path} does not exist yet")
    else:
        last = tail_lines(log_path, args.lines)
        if last:
            print(f"== Last {len(last)} lines of {log_path} ==")
            for line in last:
                print(line)
        else:
            print(f"{log_path} is empty")

    if args.follow:
        print(f"\n== Watching {log_path} for new lines ==")
        try:
            for line in follow(log_path, start_at_end=True):
                print(line)
        except KeyboardInterrupt:
            print("\nStopped watching")


if __name__ == "__main__":
    main()
