import re

def clean_text(text: str, preserve_emotion: bool = False) -> str:
    """
    Clean text while optionally preserving emotional signal.
    For public discourse, preserve_emotion=True keeps sentiment markers.
    """
    if not text:
        return ""

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    if preserve_emotion:
        # Keep ASCII + common emotional punctuation
        # Remove only truly garbage characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    else:
        # Strict clean for institutional content
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    text = text.strip()
    return text

def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    mode: str = "institutional"
) -> list[str]:
    """
    Split text into chunks.
    
    mode="institutional" — 500 word chunks, good for news articles and policy docs
    mode="public"        — 150 word chunks, preserves individual voices in forum posts
    """
    if not text:
        return []

    if mode == "public":
        chunk_size = 150
        overlap = 20

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

def detect_sentiment_strength(text: str) -> float:
    """
    Detect emotional intensity of public discourse.
    Returns 0.0 (calm) to 1.0 (highly emotional).
    Used to weight public sentiment nodes in the graph.
    """
    text_lower = text.lower()

    strong_signals = [
        "lost my job", "can't afford", "stolen", "furious", "outraged",
        "devastated", "livelihood", "unfair", "wrong", "disgusting",
        "quit", "fired", "replaced", "screwed", "fight back",
        "my work", "my art", "my music", "my voice", "without consent",
        "without permission", "never agreed", "exploitation"
    ]

    moderate_signals = [
        "concerned", "worried", "disappointed", "frustrated", "disagree",
        "don't think", "shouldn't", "should be", "need to", "have to",
        "important", "matters", "affects", "impacts"
    ]

    strong_count = sum(1 for s in strong_signals if s in text_lower)
    moderate_count = sum(1 for s in moderate_signals if s in text_lower)

    raw_score = (strong_count * 0.3) + (moderate_count * 0.1)
    return round(min(raw_score, 1.0), 3)