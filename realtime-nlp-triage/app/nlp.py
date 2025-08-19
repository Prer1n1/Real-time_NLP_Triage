from typing import Dict, Any, List, Tuple
import time
import os
import re

from langdetect import detect, DetectorFactory
from transformers import pipeline
import spacy

# Make langdetect deterministic
DetectorFactory.seed = 42

# ---------- Intent: rule-based patterns (fast, transparent) ----------
# You can tailor these to your domain. Start broad, refine later.
_INTENT_PATTERNS: List[Tuple[str, List[re.Pattern]]] = [
    ("refund_request", [
        re.compile(r"\brefund\b", re.I),
        re.compile(r"\bcharge\s*back\b", re.I),
        re.compile(r"\bmoney\s*back\b", re.I),
        re.compile(r"\breturn(ed|ing)?\b", re.I),
        re.compile(r"\bnot\s+happy\b", re.I),
    ]),
    ("order_status", [
        re.compile(r"\border\s*(#|no\.?|number)?\s*\d+", re.I),
        re.compile(r"\b(shipped|shipping|delivery|deliver|tracking)\b", re.I),
        re.compile(r"\bwhere\s+is\s+my\b", re.I),
    ]),
    ("cancel_order", [
        re.compile(r"\bcancel(l)?\b", re.I),
        re.compile(r"\bstop\s+my\s+order\b", re.I),
    ]),
    ("product_issue", [
        re.compile(r"\b(defect(ive)?|broken|damaged|doesn'?t\s+work|faulty)\b", re.I),
        re.compile(r"\bwrong\s+(item|size|color)\b", re.I),
    ]),
    ("billing_issue", [
        re.compile(r"\b(bill|billing|charged|charge|invoice|payment)\b", re.I),
        re.compile(r"\b(duplicate|double)\s+charge\b", re.I),
    ]),
    ("complaint", [
        re.compile(r"\b(terrible|awful|worst|horrible|disgusting)\b", re.I),
        re.compile(r"\bnot\s+satisfied\b", re.I),
    ]),
    ("praise", [
        re.compile(r"\b(amazing|awesome|great|excellent|love|five\s*stars|helpful)\b", re.I),
    ]),
]

# Optional: zero-shot intent (toggle on via env var USE_ZS_INTENT=1)
_ZS_MODEL = "valhalla/distilbart-mnli-12-1"  # small(ish) NLI for CPU
_ZS_LABELS = [label for (label, _) in _INTENT_PATTERNS] + ["other"]


class NLPPipeline:
    def __init__(self) -> None:
        t0 = time.perf_counter()

        # 1) Sentiment (binary, small & fast)
        self.sentiment = pipeline(
            task="sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1,
            framework="pt",
        )

        # 2) NER (spaCy small English)
        self.nlp_ner = spacy.load("en_core_web_sm", disable=["lemmatizer", "textcat"])

        # 3) Toxicity (multi-label) — Kaggle-style categories
        # Model outputs scores for: toxic, severe_toxic, obscene, threat, insult, identity_hate
        self.toxicity = pipeline(
            task="text-classification",
            model="unitary/toxic-bert",
            return_all_scores=True,
            device=-1,
        )
        # Tunable threshold: if ANY category exceeds this, mark message as toxic
        self.tox_threshold = float(os.getenv("TOX_THRESHOLD", "0.5"))

        # 4) Optional zero-shot intent (only if enabled)
        self.use_zs_intent = os.getenv("USE_ZS_INTENT", "").strip() in ("1", "true", "yes")
        self.zero_shot = None
        if self.use_zs_intent:
            self.zero_shot = pipeline(
                task="zero-shot-classification",
                model=_ZS_MODEL,
                device=-1,
            )

        self._warmup()
        self.init_ms = (time.perf_counter() - t0) * 1000.0

    def _warmup(self) -> None:
        _ = self.sentiment("Hello world!")
        _ = self.nlp_ner("Apple is looking at buying U.K. startup for $1 billion.")
        _ = self.toxicity("I dislike this.")
        if self.zero_shot:
            _ = self.zero_shot("I want a refund", _ZS_LABELS)

    # ---------- Intent helpers ----------
    def _intent_rule_based(self, text: str) -> Tuple[str, Dict[str, float]]:
        """
        Returns (best_label, scores_dict) from rule-based matches.
        Scores here are heuristic: number of pattern matches per intent (normalized).
        """
        text_norm = text or ""
        matches_per_label: Dict[str, int] = {}
        for label, patterns in _INTENT_PATTERNS:
            cnt = sum(1 for p in patterns if p.search(text_norm))
            if cnt:
                matches_per_label[label] = cnt

        if not matches_per_label:
            return "other", {}

        # Normalize scores to [0,1] by dividing by max count
        max_cnt = max(matches_per_label.values())
        scores = {k: v / max_cnt for k, v in matches_per_label.items()}
        # Pick label with highest normalized score
        best_label = max(scores, key=scores.get)
        return best_label, scores

    def _intent_zero_shot(self, text: str) -> Tuple[str, Dict[str, float]]:
        """
        Zero-shot classification over predefined labels. Returns (label, scores).
        """
        zs = self.zero_shot(text, _ZS_LABELS, multi_label=False)
        label = zs["labels"][0]
        # Normalize scores to dict {label: score}
        scores = {l: float(s) for l, s in zip(zs["labels"], zs["scores"])}
        return label, scores

    # ---------- Public API ----------
    def analyze(self, text: str) -> Dict[str, Any]:
        t0 = time.perf_counter()

        # Language
        try:
            lang = detect(text) if text and text.strip() else "unknown"
        except Exception:
            lang = "unknown"

        # Sentiment
        s0 = time.perf_counter()
        s_result = self.sentiment(text[:1024])[0]
        sentiment = {
            "label": s_result["label"].lower(),
            "score": float(s_result["score"]),
            "latency_ms": (time.perf_counter() - s0) * 1000.0,
        }

        # NER
        n0 = time.perf_counter()
        doc = self.nlp_ner(text[:2000])
        entities: List[Dict[str, Any]] = [
            {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
            for ent in doc.ents
        ]
        ner = {
            "entities": entities,
            "latency_ms": (time.perf_counter() - n0) * 1000.0,
        }

        # Toxicity
        t1 = time.perf_counter()
        tox_raw = self.toxicity(text[:1024])[0]  # list[dict]: [{'label': 'toxic', 'score': 0.xx}, ...]
        tox_scores = {item["label"]: float(item["score"]) for item in tox_raw}
        is_toxic = any(score >= self.tox_threshold for score in tox_scores.values())
        toxicity = {
            "is_toxic": bool(is_toxic),
            "scores": tox_scores,    # per-category probabilities
            "threshold": self.tox_threshold,
            "latency_ms": (time.perf_counter() - t1) * 1000.0,
        }

        # Intent (rule-based first; optionally zero-shot)
        i0 = time.perf_counter()
        label_rb, scores_rb = self._intent_rule_based(text)
        label, scores = label_rb, scores_rb

        if self.use_zs_intent and self.zero_shot:
            try:
                label_zs, scores_zs = self._intent_zero_shot(text[:512])
                # Simple tie-break: prefer zero-shot if it strongly contradicts rule-based
                # or if rule-based returned "other"
                if label_rb == "other" or (scores_zs.get(label_zs, 0) >= 0.6 and label_zs != label_rb):
                    label, scores = label_zs, scores_zs
            except Exception:
                # If zero-shot fails (e.g., download/network), fall back to rule-based
                label, scores = label_rb, scores_rb

        intent = {
            "label": label,
            "scores": scores,    # dict per label, either heuristic or zero-shot probabilities
            "method": "zero-shot" if (self.use_zs_intent and self.zero_shot) else "rules",
            "latency_ms": (time.perf_counter() - i0) * 1000.0,
        }

        total_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "language": lang,
            "sentiment": sentiment,
            "ner": ner,
            "toxicity": toxicity,
            "intent": intent,
            "total_ms": total_ms,
        }

# Singleton instance
nlp_pipeline = NLPPipeline()
