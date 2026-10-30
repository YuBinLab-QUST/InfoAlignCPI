import json
import os

BASE_DIR = "data/processed"
TRAIN_PATH = os.path.join(BASE_DIR, "train_filtered.json")
TEST_PATH  = os.path.join(BASE_DIR, "test_filtered.json")
OUT_PATH   = os.path.join(BASE_DIR, "all_cpi_raw.json")


def extract_json_objects_from_text(text: str):

    objs = []
    n = len(text)
    depth = 0
    start_idx = None
    bad = 0

    i = 0
    while i < n:
        ch = text[i]
        if ch == "{":
            if depth == 0:
                start_idx = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start_idx is not None:
                    snippet = text[start_idx:i+1]
                    try:
                        obj = json.loads(snippet)
                        objs.append(obj)
                    except Exception:
                        bad += 1
                    start_idx = None
        i += 1

    print(f"  -> extracted {len(objs)} objects, bad_snippets={bad}")
    return objs


def load_messy_big_json(path: str):

    print(f"[INFO] Try to parse large JSON from {path}")
    with open(path, "rb") as f:
        text = f.read().decode("utf-8", errors="ignore")
    objs = extract_json_objects_from_text(text)
    return objs


def main():
    train = load_messy_big_json(TRAIN_PATH)
    test  = load_messy_big_json(TEST_PATH)

    print("Train objects:", len(train))
    print("Test objects :", len(test))
    total = len(train) + len(test)
    print("Total        :", total)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(train + test, f, indent=2, ensure_ascii=False)
    print("Saved merged raw pairs to", OUT_PATH)


if __name__ == "__main__":
    main()
