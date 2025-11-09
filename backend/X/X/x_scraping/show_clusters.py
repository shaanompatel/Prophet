# show_clusters.py

import sys
import requests

ENGINE_BASE = "http://localhost:8000"


def list_markets():
    r = requests.get(f"{ENGINE_BASE}/health", timeout=10)
    r.raise_for_status()
    data = r.json()
    for m in data.get("markets", []):
        print(
            f"[{m['id']}] {m['label']} "
            f"(status={m['status']}, tweets={m['tweets']}, "
            f"weight={m['total_weight']:.1f})"
        )


def show_market(mid: int):
    r = requests.get(f"{ENGINE_BASE}/market/{mid}", timeout=10)
    if r.status_code != 200:
        print("Error:", r.status_code, r.text)
        return
    m = r.json()
    print(
        f"Market {m['id']} - {m['label']} "
        f"(status={m['status']}, tweets={m['tweets_count']})"
    )
    print("Top tweets:")
    for t in m["tweets"]:
        print(
            f"- w={t['weight']:.1f}, "
            f"likes={t['likes']}, rts={t['retweets']}, "
            f"followers={t['followers']}"
        )
        print(f"  {t['text'][:200]}")
        print()


def main():
    if len(sys.argv) == 1:
        list_markets()
    else:
        show_market(int(sys.argv[1]))


if __name__ == "__main__":
    main()
