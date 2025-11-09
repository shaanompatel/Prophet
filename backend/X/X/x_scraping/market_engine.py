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
from typing import List, Optional, Tuple

from dotenv import load_dotenv

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import requests

import asyncio
import websockets


# load envs
load_dotenv("bearerkey.env")
load_dotenv("xAIAPIKey.env")

X_BEARER_TOKEN = os.getenv("X_BEARER_TOKEN")
XAI_API_KEY = os.getenv("XAI_API_KEY")

# log files
BASE_DIR = os.path.dirname(__file__)
LOG_FILE = os.path.join(BASE_DIR, "..", "markets.log")
TWEET_LOG_FILE = os.path.join(BASE_DIR, "..", "tweets.log")

# clear logs on import (so each run starts fresh)
for path in (LOG_FILE, TWEET_LOG_FILE):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8"):
            pass
    except Exception:
        # if this fails we just skip clearing
        pass

# logging config for markets.log
logging.basicConfig(
    filename=LOG_FILE,
    filemode="w",  # overwrite on start
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

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
XAI_CHAT_URL = "https://api.x.ai/v1/chat/completions"
XAI_MODEL_NAME = "grok-4-fast"  # Grok 4 Fast

# websocket action server
ACTION_SERVER_URI = os.getenv("ACTION_SERVER_URI", "ws://localhost:8766")


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
    grok_reason_probability: Optional[str] = None

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
        self.buffer: deque[TweetRecord] = deque()
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
            text=text,
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

            for _, items in clusters.items():
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

        # filter spammy or fully past clusters
        if not self._cluster_is_suitable_for_market(recs):
            logging.info(
                "SKIP_MARKET_UNSUITABLE id=%s rough_label=%r",
                self.next_market_id,
                label_text,
            )
            return

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

        # question style title from Grok
        grok_title = self._ask_grok_for_market_title(m, recs)
        if grok_title:
            m.label = grok_title

        # resolution time from Grok, not in the past
        resolution_time, reason_for_resolution_time = self._ask_grok_for_resolution(m, recs)
        m.resolution_time = resolution_time
        m.grok_reason_resolution = reason_for_resolution_time

        # initial probability from Grok
        prob, prob_reason = self._ask_grok_for_initial_probability(m, recs)
        if prob is None:
            prob = 0.5
            prob_reason = prob_reason or "Fallback neutral prior."
        m.grok_reason_probability = prob_reason

        self.markets.append(m)

        logging.info(
            "NEW_MARKET id=%s label=%r resolution=%s prob=%.3f res_reason=%s prob_reason=%s",
            m.id,
            m.label,
            m.resolution_time.isoformat() if m.resolution_time else None,
            prob,
            (reason_for_resolution_time or "")[:200],
            (prob_reason or "")[:200],
        )

        # emit CREATE action in the same shape as the old Gemini agent
        create_action = {
            "action": "CREATE",
            "market_name": m.label,
            "probability": float(prob),
            "reason": prob_reason or "Cluster passed thresholds to become a market.",
            "market_id": m.id,
            "resolution_time": m.resolution_time.isoformat() if m.resolution_time else None,
        }
        self._emit_action(create_action)

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

    # ------------- action emission over websocket -------------

    def _emit_action(self, action: dict) -> None:
        """
        Fire and forget send of a single action JSON to the Action server.

        Matches the old Gemini decide_markets output shape:
          CREATE: {"action": "CREATE", "market_name": str, "probability": float, "reason": str}
          RESOLVE: {"action": "RESOLVE", "market_name": str, "outcome": "YES|NO", "reason": str}
        """
        try:
            asyncio.run(self._send_action_once(action))
        except RuntimeError:
            # if there is already an event loop you could swap to a queue here
            logging.warning(
                "Could not run asyncio loop for action send; skipping",
                exc_info=True,
            )
        except Exception:
            logging.warning(
                "Unexpected error while trying to emit action; skipping",
                exc_info=True,
            )

    async def _send_action_once(self, action: dict) -> None:
        try:
            async with websockets.connect(ACTION_SERVER_URI) as ws:
                msg = json.dumps(action, ensure_ascii=False)
                await ws.send(msg)
        except Exception as e:
            logging.warning("Failed to send action over websocket: %r", e)

    # ------------- Grok low level helper -------------

    def _grok_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.2,
    ) -> Optional[str]:
        """
        Call Grok 4 Fast with a system and user prompt.
        Returns the content string of the first choice, or None on error.
        """
        if not XAI_API_KEY:
            logging.warning("No XAI_API_KEY set, skipping Grok call")
            return None

        payload = {
            "model": XAI_MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        try:
            resp = requests.post(
                XAI_CHAT_URL,
                headers={
                    "Authorization": f"Bearer {XAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=30,
            )
            if resp.status_code != 200:
                logging.warning(
                    "Grok chat HTTP %s body=%r",
                    resp.status_code,
                    resp.text[:400],
                )
                resp.raise_for_status()
        except requests.RequestException as e:
            body = ""
            if hasattr(e, "response") and e.response is not None:
                body = e.response.text[:400]
            logging.warning("Grok chat HTTP error: %r body=%r", e, body)
            return None

        try:
            data = resp.json()
            choices = data.get("choices") or []
            if not choices:
                logging.warning("Grok chat response has no choices: %r", data)
                return None
            message = choices[0].get("message") or {}
            content = message.get("content")
            if not content:
                logging.warning("Grok chat first choice has no content: %r", message)
                return None
            return content
        except Exception as e:
            logging.warning(
                "Failed to parse Grok chat response: %r text=%r",
                e,
                resp.text[:400],
            )
            return None

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

    def _cluster_is_suitable_for_market(
        self,
        recs: List[TweetRecord],
    ) -> bool:
        """
        Use Grok to decide if this cluster is suitable for a public prediction market.

        Reject markets that are mostly:
          - product or brand advertising, promotions, giveaways, referral codes
          - explicit sexual content, pornography, adult services
          - any event that clearly already happened and is only being recapped
        """
        if not XAI_API_KEY:
            return True

        examples = []
        for r in recs[:8]:
            examples.append(f"- {r.text[:240]}")
        examples_text = "\n".join(examples) if examples else "- (no examples)"

        system_prompt = (
            "You classify whether a group of social media posts is suitable for a "
            "public prediction market.\n\n"
            "Reject the cluster as a market if the posts are mostly any of these:\n"
            "  - advertising, promotions, brand marketing, referral or discount codes\n"
            "  - spammy giveaways, contests to like or follow accounts\n"
            "  - explicit sexual content, pornography, escort or adult services\n"
            "  - OnlyFans style promotion or similar adult creator promotion\n"
            "  - discussion that is mainly about events that have already clearly "
            "    happened in the past, where people are just recapping, reacting, "
            "    or celebrating, with no future outcome to be decided\n\n"
            "If the posts mainly describe future or uncertain outcomes they can be "
            "suitable. If the main thing being talked about has already fully "
            "happened you must reject the cluster.\n\n"
            "Answer with a single word:\n"
            "  ACCEPT - if these posts are suitable for a public prediction market\n"
            "  REJECT - if they are mostly ads, spam, explicit adult content, or only "
            "           about past events with no remaining uncertainty.\n"
        )

        user_prompt = (
            "Here are example posts from one cluster:\n"
            f"{examples_text}\n\n"
            "Decide if this cluster is suitable for a public prediction market.\n"
            "Answer ONLY with ACCEPT or REJECT."
        )

        content = self._grok_chat(system_prompt, user_prompt, max_tokens=8)
        if not content:
            return True

        decision = content.strip().upper()
        if "REJECT" in decision:
            return False
        return True

    def _ask_grok_for_market_title(
        self,
        market: Market,
        recs: List[TweetRecord],
    ) -> Optional[str]:
        """
        Use Grok to generate a short binary prediction market question
        for this cluster, based on representative tweets.
        """
        if not XAI_API_KEY:
            return None

        examples = []
        for r in recs[:6]:
            examples.append(f"- {r.text[:240]}")
        examples_text = "\n".join(examples) if examples else "- (no examples)"

        system_prompt = (
            "You help design clear binary prediction market questions "
            "based on social media posts.\n"
            "Given example posts that describe the same real world event, "
            "write a short YES or NO question for a prediction market.\n\n"
            "Assume that any clusters that are mostly explicit adult content, "
            "advertising, or pure past event recap have already been rejected, "
            "so you only receive suitable topics here.\n\n"
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

    def _ask_grok_for_initial_probability(
        self,
        market: Market,
        recs: List[TweetRecord],
    ) -> Tuple[Optional[float], Optional[str]]:
        """
        Ask Grok for a rough initial YES probability for this market.

        Returns (probability between 0 and 1 or None, reason string).
        """
        if not XAI_API_KEY:
            return None, "No XAI_API_KEY available"

        sample_recs = sorted(recs, key=lambda r: -r.weight)[:6]
        examples = "\n".join(
            f"- {r.text[:220]}"
            for r in sample_recs
        )
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        prompt = f"""
You estimate a rough prior probability for a binary prediction market.

Market question: {market.label}
Current time (UTC): {now_iso}

Example posts that motivated this market:
{examples}

Return a JSON object with:
{{
  "probability": float between 0 and 1 for the YES outcome,
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
            prob = obj.get("probability")
            reason = obj.get("reason", "")

            try:
                if prob is None:
                    return None, reason or "Grok did not provide a probability."
                prob = float(prob)
                if not (0.0 <= prob <= 1.0):
                    return None, reason or "Grok probability out of range."
                return prob, reason
            except Exception:
                return None, reason or "Grok probability could not be parsed."
        except Exception as e:
            logging.warning("Grok probability error for market %s: %r", market.id, e)
            return None, f"error: {e!r}"

    def _ask_grok_for_resolution(
        self,
        market: Market,
        recs: List[TweetRecord],
    ) -> Tuple[Optional[datetime], str]:
        if not XAI_API_KEY:
            return None, "No XAI_API_KEY available"

        sample_recs = sorted(recs, key=lambda r: -r.weight)[:5]
        examples = "\n".join(
            f"- {r.text[:200]}"
            for r in sample_recs
        )
        now = datetime.now(timezone.utc)
        now_iso = now.strftime("%Y-%m-%dT%H:%M:%SZ")

        prompt = f"""
You are helping design prediction markets from social media chatter.

Event label: {market.label}

Example posts:
{examples}

Current time (UTC): {now_iso}

Choose a reasonable resolution time for a binary prediction market about this event.

Rules:
- If there is a clear scheduled time pick that.
- If the event is phrased as "by some date" use that date at 23:59:59 UTC.
- Never choose a resolution time in the past relative to the current time.
- If all plausible dates are in the past or timing is unclear respond with UNKNOWN.

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
            obj = json.loads(content)
            iso_dt = obj.get("iso_datetime")
            reason = obj.get("reason", "")
            if iso_dt:
                dt = datetime.fromisoformat(
                    iso_dt.replace("Z", "+00:00")
                ).astimezone(timezone.utc)
                # do not allow deadlines in the past
                if dt < now:
                    return None, reason or "Grok suggested a past datetime, treating as unknown."
                return dt, reason
            return None, reason or "Grok returned no datetime."
        except Exception as e:
            logging.warning("Grok resolution error for market %s: %r", market.id, e)
            return None, f"error: {e!r}"

    def _ask_grok_if_resolved(
        self,
        market: Market,
        recs: List[TweetRecord],
    ) -> Tuple[str, str]:
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
- "no": event clearly did not happen or was cancelled

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

                # gather tweets for this market
                recs = [r for r in self.buffer if r.assigned_market_id == m.id]
                if not recs:
                    continue

                is_past_deadline = (
                    m.resolution_time is not None and now >= m.resolution_time
                )

                if not is_past_deadline:
                    if m.last_resolution_check and (
                        now - m.last_resolution_check
                        < timedelta(minutes=RESOLUTION_CHECK_EVERY_MINUTES)
                    ):
                        continue

                    if len(recs) < MIN_MARKET_TWEETS_FOR_RESOLUTION_CHECK:
                        continue

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
                    resolve_action = {
                        "action": "RESOLVE",
                        "market_name": m.label,
                        "outcome": "YES",
                        "reason": reason or "Grok judged this event happened as expected.",
                        "market_id": m.id,
                    }
                    self._emit_action(resolve_action)

                elif status == "no":
                    m.status = "resolved_no"
                    logging.info(
                        "MARKET_RESOLVED_NO id=%s label=%r reason=%s",
                        m.id,
                        m.label,
                        (reason or "")[:200],
                    )
                    resolve_action = {
                        "action": "RESOLVE",
                        "market_name": m.label,
                        "outcome": "NO",
                        "reason": reason or "Grok judged this event did not happen.",
                        "market_id": m.id,
                    }
                    self._emit_action(resolve_action)
                # if status is "open", nothing to log or emit

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


# ------------- standalone websocket test -------------

async def _test_action_server():
    """
    Connects to the websocket action server and sends a test action.

    Run with:
        python market_engine.py

    You should see this action printed by sample_market_listener.py if the
    server and listener are wired correctly.
    """
    print(f"[test] connecting to action server at {ACTION_SERVER_URI}...")
    async with websockets.connect(ACTION_SERVER_URI) as ws:
        action = {
            "action": "TEST",
            "message": "market_engine websocket self test",
            "time": datetime.now(timezone.utc).isoformat(),
        }
        msg = json.dumps(action, ensure_ascii=False)
        print(f"[test] sending test action: {msg}")
        await ws.send(msg)
        print("[test] test action sent, closing connection.")


if __name__ == "__main__":
    # simple websocket connectivity test
    try:
        asyncio.run(_test_action_server())
    except Exception as e:
        print("[test] websocket test failed:", repr(e))
