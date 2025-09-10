import re
from dataclasses import dataclass
from typing import Optional, Tuple, Any, List
import yaml

FOOTER_PATTERNS = [
    r"View the Conversation\s*>[\s\S]*$",
    r"©\s*20\d{2}[\s\S]*$",
    r"Privacy Policy[\s\S]*$",
]


def strip_footer(text: str) -> str:
    s = text
    for pat in FOOTER_PATTERNS:
        s = re.sub(pat, "", s, flags=re.IGNORECASE | re.MULTILINE)
    return s.strip()


def fix_mojibake(s: str) -> str:
    """Attempt to fix common mojibake like â€™, â€œ, â€, Ã©, Äô etc., without external deps.
    Strategy: try latin-1 -> utf-8 roundtrip; then apply targeted replacements; finally NFKC normalize.
    """
    import unicodedata
    if not s:
        return s
    try:
        s2 = s.encode("latin-1", errors="ignore").decode("utf-8", errors="ignore")
        if s2 and ("â" in s2 or "Ã" in s2 or "Ä" in s2):
            s = s2
    except Exception:
        pass
    # Targeted replacements
    repl = {
        "â€™": "'",
        "â€˜": "'",
        "â€œ": '"',
        "â€": '"',
        "â€“": "-",
        "â€”": "-",
        "Â": "",
        "Ã©": "é",
        "Äô": "'",
        "Ã¢â‚¬â„¢": "'",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    # Also collapse the specific two-char sequence pattern 'Ä' 'ô' to apostrophe
    s = re.sub(r"Ä\s*ô", "'", s)
    # Normalize smart quotes/dashes generally
    trans = {
        0x2018: ord("'"), 0x2019: ord("'"), 0x201B: ord("'"),
        0x201C: ord('"'), 0x201D: ord('"'), 0x201E: ord('"'),
        0x2013: ord('-'), 0x2014: ord('-'),
    }
    s = s.translate(trans)
    s = unicodedata.normalize("NFKC", s)
    # Collapse excessive whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_person_name(name: str) -> str:
    """Normalize person names: trim, fix initials (A.b. -> A.B.), title-case words but keep uppercase initials.
    Also fix common host misspelling 'Bill Crystal' -> 'Bill Kristol'.
    """
    if not name:
        return name
    s = fix_mojibake(name).strip()
    if s.lower() == "bill crystal":
        return "Bill Kristol"
    parts = s.split()
    out = []
    for p in parts:
        # If looks like initial with dot(s), upper it
        if re.match(r"^[A-Za-z]\.?[A-Za-z]?\.?$", p):
            out.append(p.upper())
        else:
            # Title-case but preserve internal hyphens/periods
            out.append("-".join([sub[:1].upper()+sub[1:].lower() for sub in p.split('-')]))
    return " ".join(out)


def strip_nav_header(text: str) -> str:
    """Remove leading site navigation/header blocks like 'Home Summaries ... Download PDF ...'."""
    if not text:
        return text
    lines = text.splitlines()
    out = []
    skipping = True
    for ln in lines:
        ln_stripped = ln.strip()
        if skipping:
            if ln_stripped.lower().startswith(("home summaries", "download pdf", "guest biographies", "about us", "support our work", "contact", "search")):
                continue
            # First non-nav line flips skipping off
            skipping = False
        out.append(ln)
    return "\n".join(out).strip()


def _find_first_str(d: Any, keys: Tuple[str, ...]) -> Optional[str]:
    if isinstance(d, dict):
        for k in keys:
            if k in d and isinstance(d[k], str) and d[k].strip():
                return d[k]
        # recurse
        for v in d.values():
            val = _find_first_str(v, keys)
            if val:
                return val
    elif isinstance(d, list):
        for v in d:
            val = _find_first_str(v, keys)
            if val:
                return val
    return None


def _is_host(name: str) -> bool:
    if not name:
        return False
    n = fix_mojibake(name).strip().lower()
    return n in ("bill kristol", "bill crystal")


def _is_probable_person(name: str) -> bool:
    import re
    s = normalize_person_name(name or "")
    if not s:
        return False
    # Exclude common non-person tokens
    bad = {"filmed", "unknown", "the terror presidency"}
    if s.lower() in bad:
        return False
    # Accept names with 2-3 words starting with uppercase letters (allow hyphens and periods)
    m = re.match(r"^[A-Z][A-Za-z.'-]+( [A-Z][A-Za-z.'-]+){1,2}$", s)
    return bool(m)


def _extract_guest_from_speakers_list(items: List[Any]) -> Optional[str]:
    # Items may be list of strings or dicts with 'name' or 'speaker'
    names = []
    for it in items:
        if isinstance(it, str):
            names.append(normalize_person_name(it))
        elif isinstance(it, dict):
            for k in ("name", "speaker", "guest", "guest_name"):
                if k in it and isinstance(it[k], str) and it[k].strip():
                    names.append(normalize_person_name(it[k]))
                    break
    # pick first non-host that looks like a person
    for n in names:
        if not _is_host(n) and _is_probable_person(n):
            return n
    # if only host present, return None
    return None


def _find_guest_heuristics(data: Any, raw: str) -> Optional[str]:
    # 1) YAML containers likely: 'speakers', 'participants', 'people'
    for key in ("speakers", "participants", "people"):
        if isinstance(data, dict) and key in data and isinstance(data[key], list):
            g = _extract_guest_from_speakers_list(data[key])
            if g:
                return g
    # 2) Flatten search for a list named 'speakers' nested
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                g = _extract_guest_from_speakers_list(v)
                if g:
                    return g
    # 3) Heuristic from header text (first 15 lines)
    header = "\n".join(raw.splitlines()[:15])
    header = fix_mojibake(header)
    import re
    # Patterns: "Bill Kristol with X", "Conversation with X", "Guest: X", "Speakers: Bill Kristol, X"
    m = re.search(r"Bill\s+Kristol\s+(?:with|and)\s+([A-Z][A-Za-z .\-'']{2,})", header)
    if m:
        cand = normalize_person_name(m.group(1).strip())
        if not _is_host(cand) and _is_probable_person(cand):
            return cand
    m = re.search(r"Conversation(?:s)?\s+with\s+([A-Z][A-Za-z .\-'']{2,})", header, re.IGNORECASE)
    if m:
        cand = normalize_person_name(m.group(1).strip())
        if not _is_host(cand) and _is_probable_person(cand):
            return cand
    m = re.search(r"Guest\s*[:\-]\s*([A-Z][A-Za-z .\-'']{2,})", header, re.IGNORECASE)
    if m:
        cand = normalize_person_name(m.group(1).strip())
        if not _is_host(cand) and _is_probable_person(cand):
            return cand
    m = re.search(r"Speakers?\s*[:\-]\s*(.+)", header, re.IGNORECASE)
    if m:
        list_str = m.group(1)
        parts = re.split(r",| and ", list_str)
        for part in parts:
            cand = normalize_person_name(part.strip())
            if cand and not _is_host(cand) and _is_probable_person(cand):
                return cand
    return None


def load_yaml_transcript(path: str) -> Tuple[Optional[str], Optional[str]]:
    """Returns (text, guest_name) from a YAML transcript file.
    Tries common keys; if YAML fails to parse, treats file as plain text.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    try:
        data = yaml.safe_load(raw)
    except Exception:
        # Fallback: not valid YAML, assume raw text transcript
        return strip_footer(raw), None
    # Try to locate text and guest
    text = None
    # Prefer transcript as list of speaker turns
    if isinstance(data, dict) and isinstance(data.get("transcript"), list):
        parts = []
        for turn in data["transcript"]:
            if isinstance(turn, dict):
                spk = fix_mojibake(turn.get("speaker", "")).strip().lower()
                tx = turn.get("text")
                if not isinstance(tx, str) or not tx.strip():
                    continue
                # Skip nav/metadata speakers
                if spk in {"unknown", "filmed"}:
                    continue
                cleaned = fix_mojibake(strip_footer(strip_nav_header(tx)))
                parts.append(cleaned)
        if parts:
            text = "\n\n".join(parts)
    if not text:
        for keys in (
            ("transcript",),
            ("text",),
            ("body",),
            ("content",),
        ):
            text = _find_first_str(data, keys)
            if text:
                break
    guest = _find_first_str(data, ("guest", "guest_name", "speaker", "author"))
    # Heuristic: pick the speaker that is not Bill Kristol or 'Unknown'
    if (not guest) or _is_host(guest) or fix_mojibake(guest).strip().lower() in ("unknown", ""):
        guest = _find_guest_heuristics(data, raw)
    if text:
        text = fix_mojibake(strip_footer(strip_nav_header(text)))
        
    # Normalize guest text too
    if guest:
        guest = fix_mojibake(guest)
    return text, guest

