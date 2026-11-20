"""Text bubbles for ordinary OneBot replies and the same-session message tool."""

import re
import unicodedata

CHATPLUS_REPLY = "_qq_chatplus_reply"
QQ_USER_DEMAND = "_qq_user_demand"
QQ_ROUND_HISTORY_IDS = "_qq_round_history_ids"
TEXT_BUBBLES_SENT = "_qq_text_bubbles_sent"
LLM_REPLY_DELIVERED = "_llm_reply_delivered"
LLM_TEXT_DELIVERED = "_qq_llm_text_delivered"
QQ_REPLY_STARTED = "_qq_reply_started"
QQ_QUOTE_IF_GAP = "_qq_quote_if_gap"
NO_REPLY = "<no_reply/>"

SENTENCE_END = re.compile(r'[。！？!?]+[”’」』）》】]*|[\r\n]+')
CLAUSE_SEPARATOR = re.compile(r"[，；;]|(?<!\d),|,(?!\d)")
SOFT_BREAK = re.compile(r"[，、；;]|\s")
INLINE_LITERAL = re.compile(r"https?://[^\s，。！？；]+|\x60[^\x60\r\n]+\x60")
UNBROKEN_WORD = re.compile(r"[\w./%:@+#=-]+", re.ASCII)
LONG_BUBBLE = 60
QQ_EMOJI_MARKER = re.compile(r"\[emoji[ \t]+(?:name|tag)[ \t]*=[^\[\]\r\n]*\]", re.I)
CHINESE_PAUSE = re.compile(
    r"(?<=[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff])[ \t\u00a0\u3000]+"
    r"(?=[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff])"
)


def _outside_literals(pattern, text):
    literals = iter(INLINE_LITERAL.finditer(text))
    literal = next(literals, None)
    for match in pattern.finditer(text):
        while literal and literal.end() <= match.start():
            literal = next(literals, None)
        if not literal or match.start() < literal.start():
            yield match


def strip_qq_emoji_markers(text):
    """Remove unrendered QQ emoji markup from ordinary text."""
    if "\x60\x60\x60" in text:
        return text
    start = 0
    parts = []
    for marker in _outside_literals(QQ_EMOJI_MARKER, text):
        parts.append(text[start:marker.start()])
        start = marker.end()
    parts.append(text[start:])
    return "".join(parts)


def _punctuate_chinese_pauses(text):
    start = 0
    parts = []
    for gap in _outside_literals(CHINESE_PAUSE, text):
        parts.extend((text[start:gap.start()], "，"))
        start = gap.end()
    parts.append(text[start:])
    return "".join(parts)


def _sentences(text):
    start = 0
    for end in _outside_literals(SENTENCE_END, text):
        punctuation = end.group().replace("。", "").strip()
        yield text[start:end.start()] + punctuation
        start = end.end()
    if start < len(text):
        yield text[start:]


def _clauses(text):
    start = 0
    comma_seen = False
    for separator in _outside_literals(CLAUSE_SEPARATOR, text):
        if separator.group() in "；;" or comma_seen:
            yield text[start:separator.start()]
            start = separator.end()
            comma_seen = False
        else:
            comma_seen = True
    yield text[start:]


def _shorten(text):
    text = text.strip()
    while len(text) > LONG_BUBBLE:
        # Prefer a readable break before the fallback limit.
        soft = next(
            (match for match in reversed(list(_outside_literals(SOFT_BREAK, text)))
             if LONG_BUBBLE // 2 <= match.start() <= LONG_BUBBLE),
            None,
        )
        if soft:
            yield text[:soft.start()].strip()
            text = text[soft.end():].strip()
            continue
        cut = LONG_BUBBLE
        for pattern in (INLINE_LITERAL, UNBROKEN_WORD):
            for item in pattern.finditer(text):
                if item.start() < cut < item.end():
                    cut = item.start() or item.end()
                    break
        yield text[:cut].strip()
        text = text[cut:].strip()
    if text:
        yield text


def split_qq_text(text):
    """Keep sentence breaks as separate sends, including in long replies."""
    text = strip_qq_emoji_markers(text).strip()
    if not text:
        return []
    if "\x60\x60\x60" in text:
        return [text]
    return [
        bubble
        for sentence in _sentences(_punctuate_chinese_pauses(text))
        for clause in _clauses(sentence)
        for bubble in _shorten(clause)
    ]


def suppress_qq_final(event, text):
    """Keep an explicit ChatPlus silence decision from being sent as text."""
    return bool(event.get_extra(CHATPLUS_REPLY) and text.strip().casefold() == NO_REPLY)


def _qq_text_key(text):
    return "".join(
        c for c in text if not c.isspace() and not unicodedata.category(c).startswith("P")
    )


def undelivered_qq_text(text, delivered):
    """Drop an already delivered prefix, even when the model changed its punctuation."""
    previous = [_qq_text_key(part) for part in delivered]
    if not any(previous) or not text:
        return text
    candidates = sorted({key for key in [*previous, "".join(previous)] if key}, key=len, reverse=True)
    while text:
        normalized = _qq_text_key(text)
        removed = False
        for prefix in candidates:
            if not normalized.startswith(prefix):
                continue
            positions = [
                i for i, c in enumerate(text)
                if not c.isspace() and not unicodedata.category(c).startswith("P")
            ]
            cut = positions[len(prefix) - 1] + 1
            if cut < len(text) and (
                not text[cut].isspace() and not unicodedata.category(text[cut]).startswith("P")
            ):
                continue
            text = text[cut:].lstrip(" \t\r\n，,。！？!?；;、：:")
            removed = True
            break
        if not removed:
            break
    return text


def _model_error_class(diagnostic: object) -> str:
    """Classify only the native runner's error envelope for trace observations."""
    prefix = "All chat models failed: "
    if not isinstance(diagnostic, str) or not diagnostic.startswith(prefix):
        return "unknown"
    name, separator, detail = diagnostic.removeprefix(prefix).partition(":")
    if not separator:
        return "unknown"

    known = {
        "APITimeoutError": "timeout",
        "TimeoutError": "timeout",
        "APIConnectionError": "connection",
        "ConnectionError": "connection",
        "RateLimitError": "rate_limit",
        "InternalServerError": "server_error",
        "AuthenticationError": "authentication",
        "PermissionDeniedError": "permission_denied",
    }
    if name in known:
        return known[name]

    status = 400 if name == "BadRequestError" else None
    if name == "APIStatusError":
        match = re.match(r"\s*Error code:\s*(\d{3})\b", detail)
        if match:
            status = int(match.group(1))
    if status == 400:
        detail = detail.casefold()
        if "input text data may contain inappropriate content" in detail:
            return "content_policy"
        return "bad_request"
    if status is not None and 500 <= status < 600:
        return "server_error"
    return {
        401: "authentication", 403: "permission_denied",
        408: "timeout", 429: "rate_limit",
    }.get(status, "unknown")


def _free_tier_exhausted(detail: str) -> bool:
    return (
        re.search(r"\ballocationquota\.freetieronly\b(?!\.\w)", detail) is not None
        or "the free tier of the model has been exhausted" in detail
    )


def _rate_limit_sub_kind(diagnostic: str) -> str:
    """Use explicit evidence for the kind of 429 without inferring retryability."""
    detail = diagnostic.casefold()
    quota = _free_tier_exhausted(detail) or any(
        text in detail for text in (
            "free allocated quota exceeded",
            "your token-plan quota has been exhausted",
        )
    )
    throttle = re.search(
        r"\b(?:throttling\.(?:ratequota|burstrate|concurrency|serviceoverloaded|resourceexhausted)"
        r"|limit_requests|limit_burst_rate|resourceexhausted)\b(?!\.\w)",
        detail,
    ) is not None or any(
        text in detail for text in (
            "requests rate limit exceeded",
            "you have exceeded your request limit",
            "too many requests",
            "allocated quota exceeded, please increase your quota limit",
            "too many concurrent requests",
            "requests throttling triggered",
        )
    )
    if quota == throttle:
        return "unknown"
    return "quota" if quota else "throttle"


def model_error_trace_fields(diagnostic: object) -> dict[str, str]:
    """Return bounded observation labels for a runner result or an exception."""
    if isinstance(diagnostic, Exception):
        diagnostic = f"All chat models failed: {type(diagnostic).__name__}: {diagnostic}"
    error_class = _model_error_class(diagnostic)
    fields = {"error_class": error_class}
    if isinstance(diagnostic, str):
        if error_class == "rate_limit":
            fields["sub_kind"] = _rate_limit_sub_kind(diagnostic)
        elif error_class == "permission_denied" and _free_tier_exhausted(diagnostic.casefold()):
            fields["sub_kind"] = "quota"
    return fields
