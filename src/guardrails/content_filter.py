# src/guardrails/content_filter.py

# A simple list of sensitive topics to filter.
# This is a basic implementation and should be expanded for a real application.
SENSITIVE_KEYWORDS = [
    "violence", "hate speech", "self-harm", "explicit content",
    "bạo lực", "tự tử", "nội dung người lớn", "thù ghét"
]

def is_sensitive_content(query: str) -> bool:
    """
    Checks for sensitive content using a simple keyword match.
    Returns True if a keyword is found, False otherwise.
    """
    lower_query = query.lower()
    for keyword in SENSITIVE_KEYWORDS:
        if keyword in lower_query:
            print(f"⚠️ Sensitive content keyword detected: '{keyword}'")
            return True
    return False
