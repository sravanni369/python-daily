"""Day 14 — What it costs an LLM to say "Tesla" in the markets where Tesla sells.

Source: "Hands-On Large Language Models", Alammar & Grootendorst (O'Reilly).
  p. 74 — GPT models tokenize with byte pair encoding.
  p. 78 — GPT-2 and RoBERTa "include bytes as tokens in their vocabulary as the final
          building block to fall back to when they encounter characters they can't
          otherwise represent."
  p. 82 — GPT-2, 2019, BPE, vocabulary size 50,257.

That fallback is the whole story. A script the tokenizer has no merges for gets billed as raw
UTF-8 bytes, and UTF-8 is not equal: Latin is 1 byte per character, Devanagari, Telugu, Chinese
and Japanese are 3. Same company, same sentence, more units billed.

Byte length is the worst case, not every tokenizer's behaviour for every script. It is the
ceiling, and it is deterministic, which is why it is worth measuring.

Standard library only. No pip, no network.
"""

from typing import Dict

# Standard market forms of each name. Transliterations, not invented translations.
COMPANIES: Dict[str, Dict[str, str]] = {
    "Tesla":     {"English": "Tesla", "Chinese": "特斯拉", "Japanese": "テスラ",
                  "Hindi": "टेस्ला", "Telugu": "టెస్లా", "Russian": "Тесла"},
    "SpaceX":    {"English": "SpaceX", "Chinese": "太空探索技术公司", "Japanese": "スペースX",
                  "Hindi": "स्पेसएक्स", "Telugu": "స్పేస్‌ఎక్స్", "Russian": "СпейсИкс"},
    "Neuralink": {"English": "Neuralink", "Chinese": "神经链接", "Japanese": "ニューラリンク",
                  "Hindi": "न्यूरालिंक", "Telugu": "న్యూరాలింక్", "Russian": "Нейралинк"},
    "xAI":       {"English": "xAI", "Chinese": "xAI", "Japanese": "xAI",
                  "Hindi": "xAI", "Telugu": "xAI", "Russian": "xAI"},
}

PRICE_PER_MILLION = 3.00   # USD, a stated rate. Substitute your provider's.


def byte_cost(text: str) -> int:
    """Bytes billed if the tokenizer falls back to raw UTF-8. The ceiling."""
    return len(text.encode("utf-8"))


def report(company: str, forms: Dict[str, str]) -> None:
    base = byte_cost(forms["English"])
    print(f"\n{company}")
    print(f"  {'market':<10} {'written':<18} {'bytes':>6} {'vs English':>11}")
    for market, name in forms.items():
        b = byte_cost(name)
        print(f"  {market:<10} {name:<18} {b:>6} {b / base:>10.2f}x")
    worst = max(forms.values(), key=byte_cost)
    ratio = byte_cost(worst) / base
    if ratio == 1.0:
        print("  -> written identically everywhere. No penalty. This is the exception.")
    else:
        extra = (byte_cost(worst) - base) * 1_000_000 / 1_000_000 * PRICE_PER_MILLION
        print(f"  -> worst case {ratio:.2f}x English. One million mentions costs "
              f"${extra:,.2f} more, for the same word.")


if __name__ == "__main__":
    print("Byte-fallback cost of Musk company names, by market script")
    print(f"Priced at ${PRICE_PER_MILLION:.2f} per million tokens\n" + "=" * 62)
    for company, forms in COMPANIES.items():
        report(company, forms)

    print("\n" + "=" * 62)
    all_names = [n for f in COMPANIES.values() for n in f.values()]
    latin = [n for n in all_names if byte_cost(n) == len(n)]
    print(f"{len(all_names)} names measured. {len(latin)} cost 1 byte per character. "
          f"{len(all_names) - len(latin)} cost more.")
