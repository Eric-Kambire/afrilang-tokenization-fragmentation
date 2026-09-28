import argparse, os
from pathlib import Path
import sentencepiece as spm

def train(input_file, out_dir, vocab_size=2000):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    for model_type in ["bpe", "unigram"]:
        prefix = str(out / model_type)
        spm.SentencePieceTrainer.train(
            input=input_file,
            model_prefix=prefix,
            vocab_size=vocab_size,
            model_type=model_type,
            character_coverage=1.0,
            hard_vocab_limit=False,
            byte_fallback=False,
            unk_id=0, bos_id=1, eos_id=2, pad_id=3
        )
        print("trained", model_type, prefix + ".model")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out_dir", default="artifacts/tokenizers")
    ap.add_argument("--vocab_size", type=int, default=2000)
    args = ap.parse_args()
    train(args.input, args.out_dir, args.vocab_size)
