import json
import requests
from pathlib import Path


# Code was copied from https://github.com/rasbt/math_full_minus_math500
def load_math_data(cache_path: str = "math_full_minus_math500.json"):
    local_path = Path(cache_path)
    url = (
        "https://raw.githubusercontent.com/rasbt/math_full_minus_math500/"
        "main/math_full_minus_math500.json"
    )

    if local_path.exists():
        with local_path.open("r", encoding="utf-8") as f:
            math_data = json.load(f)
    else:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        math_data = r.json()

        # save locally
        with local_path.open("w", encoding="utf-8") as f:
            json.dump(math_data, f, ensure_ascii=False, indent=2)

    return math_data
