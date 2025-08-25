from typing import List

def make_kmers(seq: str, k: int = 6) -> List[str]:
    """Return overlapping k-mers from a DNA sequence (A/C/G/T/N only)."""
    if not seq:
        return []
    seq = seq.strip().upper()
    allowed = set("ACGTN")
    seq = "".join(c for c in seq if c in allowed)
    if len(seq) < k:
        return []
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]
