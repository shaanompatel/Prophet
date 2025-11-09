# market_engine.py

import os
import math
import json
import time
import logging
import threading
from collections import deque, Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import List, Optional

from dotenv import load_dotenv


import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import requests


load_dotenv("bearerkey.env")
load_dotenv("xAIAPIKEY.env")
LOG_FILE = os.path.join(os.path.dirname(__file__), "..", "markets.log")
TWEET_LOG_FILE = os.path.join(os.path.dirname(__file__), "..", "tweets.log")


# sampled stream buffer window
BUFFER_WINDOW_MINUTES = 60

# clustering thresholds
ASSIGN_THRESHOLD = 0.55      # similarity needed to join existing market
MIN_CLUSTER_POINTS = 10      # min orphan tweets to consider a new market
MIN_CLUSTER_TOTAL_WEIGHT = 150.0

# resolution thresholds
MIN_MARKET_TWEETS_FOR_RESOLUTION_CHECK = 15
RESOLUTION_CHECK_EVERY_MINUTES = 10

# xAI
XAI_API_KEY = os.getenv("XAI_API_KEY")
XAI_CHAT_URL = "https://api.x.ai/v1/chat/completions"
XAI_MODEL_NAME = "grok-4-fast"  # latest cheap reasoning model


logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


@dataclass
class TweetRecord:
    id: str
    text: str
    created_at: datetime
    user_id: str
    followers: int
    verified: bool
    likes: int
    retweets: int
    replies: int
    quotes: int
    weight: float
    embedding: np.ndarray
    assigned_market_id: Optional[int] = None


@dataclass
class Market:
    id: int
    label: str
    centroid: np.ndarray
    total_weight: float = 0.0
    tweet_ids: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolution_time: Optional[datetime] = None
    status: str = "open"  # "open", "resolved_yes", "resolved_no"
    yes_weight: float = 0.0
    no_weight: float = 0.0
    last_resolution_check: Optional[datetime] = None
    grok_reason_resolution: Optional[str] = None
    grok_reason_status: Optional[str] = None

    def add_tweet(self, rec: TweetRecord):
        # weighted centroid update
        if self.total_weight <= 0:
            self.centroid = rec.embedding.astype("float32")
        else:
            self.centroid = (
                self.centroid * self.total_weight + rec.embedding * rec.weight
            ) / (self.total_weight + rec.weight)
        self.total_weight += rec.weight
        self.tweet_ids.append(rec.id)


class MarketEngine:
    def __init__(self):
        self.lock = threading.Lock()
        self.buffer = deque()
        self.markets: List[Market] = []
        self.next_market_id = 0
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self._last_cluster_time = datetime.now(timezone.utc)

        # path for tweet log
        self.tweet_log_path = TWEET_LOG_FILE

    # ------------- tweet ingestion -------------

    def ingest_tweet(self, tweet: dict, source: str = "stream") -> None:
        """
        Normalize a raw tweet dict, compute embedding and weight,
        create a TweetRecord and push it into the buffer.
        """
        created = tweet["created_at"]
        if isinstance(created, str):
            created = datetime.fromisoformat(
                created.replace("Z", "+00:00")
            ).astimezone(timezone.utc)

        text = tweet.get("text") or ""
        emb = self.embedder.encode([text])[0].astype("float32")
        wt = self._tweet_weight(tweet, created)

        rec = TweetRecord(
            id=str(tweet["id"]),
            text=text,  # <- use the local variable `text`, not rec.text
            created_at=created,
            user_id=str(tweet.get("user_id", "")),
            followers=int(tweet.get("followers", 0)),
            verified=bool(tweet.get("verified", False)),
            likes=int(tweet.get("likes", 0)),
            retweets=int(tweet.get("retweets", 0)),
            replies=int(tweet.get("replies", 0)),
            quotes=int(tweet.get("quotes", 0)),
            weight=wt,
            embedding=emb,
        )

        with self.lock:
            self._drop_old_locked()
            self._assign_or_orphan_locked(rec)
            self.buffer.append(rec)
            # log after assignment so assigned_market_id is filled if any
            self._log_tweet(rec, source=source)



    def _drop_old_locked(self):
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(minutes=BUFFER_WINDOW_MINUTES)
        while self.buffer and self.buffer[0].created_at < cutoff:
            self.buffer.popleft()

    def _tweet_weight(self, tweet: dict, created_at: datetime) -> float:
        likes = int(tweet.get("likes", 0))
        retweets = int(tweet.get("retweets", 0))
        replies = int(tweet.get("replies", 0))
        quotes = int(tweet.get("quotes", 0))
        followers = int(tweet.get("followers", 0))
        verified = bool(tweet.get("verified", False))

        base = likes + 2 * retweets + replies + 2 * quotes
        verified_bonus = 20 if verified else 0

        minutes = max(
            0.0,
            (datetime.now(timezone.utc) - created_at).total_seconds() / 60.0,
        )
        half_life = 120.0
        decay = 0.5 ** (minutes / half_life)

        score = (
            math.log1p(base) * 2.0
            + math.log1p(max(followers, 0) + 1) * 1.5
            + verified_bonus
        )
        return score * decay

    # ------------- market assignment -------------

    def _assign_or_orphan_locked(self, rec: TweetRecord) -> None:
        if not self.markets:
            # everything is orphan until clustering promotes clusters
            return

        sims = [self._cosine_sim(rec.embedding, m.centroid) for m in self.markets]
        best_idx = int(np.argmax(sims))
        best_sim = sims[best_idx]

        if best_sim >= ASSIGN_THRESHOLD:
            market = self.markets[best_idx]
            market.add_tweet(rec)
            rec.assigned_market_id = market.id

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        a_norm = a / (np.linalg.norm(a) + 1e-9)
        b_norm = b / (np.linalg.norm(b) + 1e-9)
        return float(np.dot(a_norm, b_norm))

    # ------------- clustering for new markets -------------

    def maybe_cluster_orphans(self) -> None:
        with self.lock:
            now = datetime.now(timezone.utc)
            if now - self._last_cluster_time < timedelta(minutes=2):
                return
            self._last_cluster_time = now

            orphans = [r for r in self.buffer if r.assigned_market_id is None]
            if len(orphans) < MIN_CLUSTER_POINTS:
                return

            embs = np.stack([r.embedding for r in orphans])
            weights = np.array([r.weight for r in orphans])
            k = max(2, min(10, len(orphans) // MIN_CLUSTER_POINTS))
            km = KMeans(n_clusters=k, random_state=0, n_init="auto")
            labels = km.fit_predict(embs)

            clusters = {}
            for label, rec, w in zip(labels, orphans, weights):
                clusters.setdefault(label, []).append((rec, w))

            for label, items in clusters.items():
                recs = [r for (r, _) in items]
                total_w = sum(w for (_, w) in items)
                if len(items) < MIN_CLUSTER_POINTS or total_w < MIN_CLUSTER_TOTAL_WEIGHT:
                    continue

                self._promote_cluster_locked(recs)

    def _promote_cluster_locked(self, recs: List[TweetRecord]) -> None:
        embs = np.stack([r.embedding for r in recs])
        weights = np.array([r.weight for r in recs])
        centroid = np.average(embs, axis=0, weights=weights).astype("float32")

        label_text = self._label_from_cluster(recs)

        m = Market(
            id=self.next_market_id,
            label=label_text,
            centroid=centroid,
        )
        self.next_market_id += 1

        # assign tweets to this market
        for r in recs:
            m.add_tweet(r)
            r.assigned_market_id = m.id

        # ask Grok for a better question style title
        grok_title = self._ask_grok_for_market_title(m, recs)
        if grok_title:
            m.label = grok_title

        # ask Grok for a resolution time
        resolution_time, reason = self._ask_grok_for_resolution(m, recs)
        m.resolution_time = resolution_time
        m.grok_reason_resolution = reason

        self.markets.append(m)

        logging.info(
            "NEW_MARKET id=%s label=%r resolution=%s reason=%s",
            m.id,
            m.label,
            m.resolution_time.isoformat() if m.resolution_time else None,
            (reason or "")[:200],
        )

    def _log_tweet(self, rec: TweetRecord, source: str = "unknown"):
        """
        Append a JSON line describing this tweet to tweets.log
        """
        entry = {
            "id": rec.id,
            "text": rec.text,
            "created_at": rec.created_at.isoformat(),
            "source": source,
            "user_id": rec.user_id,
            "followers": rec.followers,
            "verified": rec.verified,
            "likes": rec.likes,
            "retweets": rec.retweets,
            "replies": rec.replies,
            "quotes": rec.quotes,
            "weight": float(rec.weight),
            "assigned_market_id": rec.assigned_market_id,
        }
        try:
            with open(self.tweet_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logging.warning("Failed to write tweet log: %r", e)


    @staticmethod
    def _label_from_cluster(recs: List[TweetRecord]) -> str:
        hashtags = Counter()
        words = Counter()
        for r in recs:
            for tok in r.text.split():
                t = tok.lower()
                if t.startswith("#"):
                    hashtags[t] += 1
                elif t.isalpha() and len(t) > 3:
                    words[t] += 1
        top_tags = [t for (t, _) in hashtags.most_common(3)]
        top_words = [w for (w, _) in words.most_common(5) if w not in top_tags]
        if top_tags:
            return " / ".join(top_tags)
        if top_words:
            return " ".join(top_words[:3])
        return "misc topic"

    # ------------- Grok helpers -------------
    def _ask_grok_for_market_title(
        self,
        market: "Market",
        recs: List["TweetRecord"],
    ) -> Optional[str]:
        """
        Use Grok to generate a short binary prediction market question
        for this cluster, based on representative tweets.
        """
        if not XAI_API_KEY:
            return None

        # take some representative tweets, assume recs are already for this cluster
        examples = []
        for r in recs[:6]:
            examples.append(f"- {r.text[:240]}")
        examples_text = "\n".join(examples) if examples else "- (no examples)"

        system_prompt = (
            "You help design clear binary prediction market questions "
            "based on social media posts.\n"
            "Given example posts that describe the same real world event, "
            "write a short YES or NO question for a prediction market.\n\n"
            "Guidelines:\n"
            "  - Make it specific and answerable.\n"
            "  - Add a time bound if the posts strongly imply one "
            "    for example 'by 2025 11 20' or 'before the end of 2025'.\n"
            "  - Prefer forms like 'Will X happen by Y' when natural, "
            "    but do not force that shape if it sounds wrong.\n"
            "  - Avoid vague wording such as 'soon' or 'in the near future'.\n"
            "  - Output only the question text, no explanation.\n"
        )

        user_prompt = (
            f"Internal cluster id: {market.id}\n"
            f"Current rough label: {market.label!r}\n\n"
            f"Example posts about this event:\n"
            f"{examples_text}\n\n"
            "Now write a concise binary prediction market question for this event.\n"
            "Answer ONLY with the question text, without quotes."
        )

        try:
            # reuse your existing Grok chat helper
            content = self._grok_chat(system_prompt, user_prompt, max_tokens=64)
            if not content:
                return None
            title = content.strip().strip('"').strip("'")
            if len(title) < 8:
                return None
            return title
        except Exception as e:
            logging.warning("Grok title generation failed: %r", e)
            return None


    def _ask_grok_for_resolution(
        self,
        market: Market,
        recs: List[TweetRecord],
    ):
        if not XAI_API_KEY:
            return None, "No XAI_API_KEY available"

        sample_recs = sorted(recs, key=lambda r: -r.weight)[:5]
        examples = "\n".join(
            f"- {r.text[:200]}"
            for r in sample_recs
        )
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        prompt = f"""
You are helping design prediction markets from social media chatter.

Event label: {market.label}

Example posts:
{examples}

Current time (UTC): {now_iso}

Choose a reasonable resolution time for a binary prediction market about this event.
If there is a clear scheduled time, pick that.
If it is phrased as "by some date", use that date at 23:59:59 UTC.
If there is not enough info, respond with UNKNOWN.

Answer as strict JSON:
{{
  "iso_datetime": "YYYY-MM-DDTHH:MM:SSZ" or null,
  "choice": "scheduled" or "deadline" or "unknown",
  "reason": "short explanation"
}}
""".strip()

        try:
            headers = {
                "Authorization": f"Bearer {XAI_API_KEY}",
                "Content-Type": "application/json",
            }
            body = {
                "model": XAI_MODEL_NAME,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a precise assistant that outputs only valid JSON.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "stream": False,
            }
            resp = requests.post(
                XAI_CHAT_URL,
                headers=headers,
                json=body,
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            # try to parse JSON from content
            obj = json.loads(content)
            iso_dt = obj.get("iso_datetime")
            reason = obj.get("reason", "")
            if iso_dt:
                dt = datetime.fromisoformat(
                    iso_dt.replace("Z", "+00:00")
                ).astimezone(timezone.utc)
                return dt, reason
            return None, reason or "Grok returned no datetime"
        except Exception as e:
            logging.warning("Grok resolution error for market %s: %r", market.id, e)
            return None, f"error: {e!r}"

    def _ask_grok_if_resolved(
        self,
        market: Market,
        recs: List[TweetRecord],
    ):
        if not XAI_API_KEY:
            return "open", "No XAI_API_KEY available"

        sample_recs = sorted(recs, key=lambda r: -r.weight)[:10]
        examples = "\n".join(
            f"- {r.text[:220]}"
            for r in sample_recs
        )
        res_time_str = (
            market.resolution_time.strftime("%Y-%m-%dT%H:%M:%SZ")
            if market.resolution_time
            else "unknown"
        )
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        prompt = f"""
You judge whether a prediction market should now be resolved based on social media posts.

Market title: {market.label}
Planned resolution time (UTC): {res_time_str}
Current time (UTC): {now_iso}

Example posts that are about this market:
{examples}

Is the event clearly already resolved?
Answer as JSON with one of three statuses:

- "open": event outcome is not clear yet
- "yes": event clearly happened as the market expected
- "no": event clearly did not happen, or was cancelled

Return strict JSON:
{{
  "status": "open" or "yes" or "no",
  "reason": "short explanation"
}}
""".strip()

        try:
            headers = {
                "Authorization": f"Bearer {XAI_API_KEY}",
                "Content-Type": "application/json",
            }
            body = {
                "model": XAI_MODEL_NAME,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a precise assistant that outputs only valid JSON.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "stream": False,
            }
            resp = requests.post(
                XAI_CHAT_URL,
                headers=headers,
                json=body,
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            obj = json.loads(content)
            status = obj.get("status", "open")
            reason = obj.get("reason", "")
            if status not in {"open", "yes", "no"}:
                status = "open"
            return status, reason
        except Exception as e:
            logging.warning("Grok status error for market %s: %r", market.id, e)
            return "open", f"error: {e!r}"

    # ------------- resolution checks -------------

    def maybe_check_resolutions(self) -> None:
        now = datetime.now(timezone.utc)
        with self.lock:
            for m in self.markets:
                if m.status != "open":
                    continue

                if m.last_resolution_check and (
                    now - m.last_resolution_check
                    < timedelta(minutes=RESOLUTION_CHECK_EVERY_MINUTES)
                ):
                    continue

                # gather tweets for this market
                recs = [r for r in self.buffer if r.assigned_market_id == m.id]
                if len(recs) < MIN_MARKET_TWEETS_FOR_RESOLUTION_CHECK:
                    continue

                # simple rule: only check after planned resolution time, if we have one
                if m.resolution_time and now < m.resolution_time:
                    continue

                m.last_resolution_check = now
                status, reason = self._ask_grok_if_resolved(m, recs)
                m.grok_reason_status = reason

                if status == "yes":
                    m.status = "resolved_yes"
                    logging.info(
                        "MARKET_RESOLVED_YES id=%s label=%r reason=%s",
                        m.id,
                        m.label,
                        (reason or "")[:200],
                    )
                elif status == "no":
                    m.status = "resolved_no"
                    logging.info(
                        "MARKET_RESOLVED_NO id=%s label=%r reason=%s",
                        m.id,
                        m.label,
                        (reason or "")[:200],
                    )
                # if open, nothing to log

    # ------------- utilities for debug -------------

    def snapshot(self):
        with self.lock:
            return {
                "buffer_size": len(self.buffer),
                "markets": [
                    {
                        "id": m.id,
                        "label": m.label,
                        "status": m.status,
                        "total_weight": m.total_weight,
                        "resolution_time": (
                            m.resolution_time.isoformat()
                            if m.resolution_time
                            else None
                        ),
                        "tweets": len(m.tweet_ids),
                    }
                    for m in self.markets
                ],
            }
