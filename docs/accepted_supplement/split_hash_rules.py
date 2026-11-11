"""Original CARGO tile-ID rules; no data loading, training, or inference."""
import hashlib
def holdout(sample_id, seed=3407, fraction=.05):
    payload=f"{seed}|{sample_id}|p27-holdout|0|split".encode("utf-8")
    value=int.from_bytes(hashlib.sha256(payload).digest()[:8],"little")%(2**31)
    return value/float(2**31)<fraction
def selector_role(sample_id, seed=3407):
    if holdout(sample_id,seed):return "risk_calibration"
    payload=f"{seed}|p27-selector-split|{sample_id}".encode("utf-8")
    value=int.from_bytes(hashlib.sha256(payload).digest()[:8],"little")/float(2**64)
    return "early_stop" if value<.15 else "fit"
