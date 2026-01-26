# audit_index.py
from rag.build_index import load_index
import re

TARGET = "ADM-1.5.3"

# Common variants to detect
VARIANTS = [
    r"ADM[-–]\s*1\.5\.3",   # ADM-1.5.3 or ADM–1.5.3
    r"ADM\s+1\.5\.3",       # ADM 1.5.3
    r"ADM[-–]\s*1\s*[\.\-–]\s*5\s*[\.\-–]\s*3",  # ADM-1-5-3 variants
]

def normalize(s: str) -> str:
    return " ".join((s or "").split())

def main():
    index = load_index()
    docstore = index.docstore

    total_chunks = 0
    exact_hits = []
    variant_hits = []

    for node_id, node in docstore.docs.items():
        total_chunks += 1
        text = node.get_text()
        norm = normalize(text)

        # Exact match
        if TARGET in norm:
            exact_hits.append((node_id, norm))

        # Variant match
        for pat in VARIANTS:
            if re.search(pat, norm):
                variant_hits.append((node_id, pat, norm))
                break

    print("\n=== INDEX AUDIT RESULTS ===")
    print(f"Total chunks: {total_chunks}")
    print(f"Exact '{TARGET}' hits: {len(exact_hits)}")
    print(f"Variant hits: {len(variant_hits)}")

    if exact_hits:
        print("\n--- Example exact hit (first 300 chars) ---")
        print(exact_hits[0][1][:300])

    if not exact_hits and variant_hits:
        print("\n--- Example variant hit (first 300 chars) ---")
        print(variant_hits[0][2][:300])
        print("Matched pattern:", variant_hits[0][1])

    if not exact_hits and not variant_hits:
        print("\n❌ No exact or variant matches found.")
        print("Likely causes:")
        print("- Header lines dropped during chunking")
        print("- ID split across chunks")
        print("- OCR / normalization altered identifiers")

if __name__ == "__main__":
    main()
