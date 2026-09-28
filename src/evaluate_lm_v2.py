import argparse
import json
import math
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn.functional as F

from train_tiny_lm import TinyTransformerLM


def load_model(checkpoint_path, vocab_size, device):
    model = TinyTransformerLM(vocab=vocab_size).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt

    model.load_state_dict(state)
    model.eval()

    return model


@torch.no_grad()
def evaluate(model, ids, seq_len, device):
    total_nll = 0.0
    total_predicted = 0

    # Includes the final partial sequence.
    for start in range(0, len(ids) - 1, seq_len):
        end = min(start + seq_len + 1, len(ids))
        chunk = ids[start:end]

        if len(chunk) < 2:
            continue

        x = torch.tensor(
            chunk[:-1],
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)

        y = torch.tensor(
            chunk[1:],
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)

        logits = model(x)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            reduction="sum",
        )

        total_nll += loss.item()
        total_predicted += y.numel()

    return total_nll, total_predicted


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--text", required=True)
    parser.add_argument("--spm-model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default=None)

    args = parser.parse_args()

    device = args.device or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    text = Path(args.text).read_text(encoding="utf-8")

    sp = spm.SentencePieceProcessor()
    sp.load(args.spm_model)

    ids = sp.encode(text, out_type=int)

    if len(ids) < 2:
        raise ValueError("Encoded text contains fewer than two tokens.")

    model = load_model(
        args.checkpoint,
        sp.get_piece_size(),
        device,
    )

    total_nll, predicted_tokens = evaluate(
        model,
        ids,
        args.seq_len,
        device,
    )

    tokenizer_tokens = len(ids)
    chars = len(text)
    byte_count = len(text.encode("utf-8"))

    nll_per_token = total_nll / predicted_tokens
    bits_per_token = nll_per_token / math.log(2.0)

    total_bits = total_nll / math.log(2.0)

    bpc = total_bits / chars
    bpb = total_bits / byte_count

    result = {
        "tokenizer_tokens": tokenizer_tokens,
        "predicted_tokens": predicted_tokens,
        "chars": chars,
        "bytes": byte_count,
        "total_nll": total_nll,
        "nll_per_token": nll_per_token,
        "bits_per_token": bits_per_token,
        "bpc": bpc,
        "bpb": bpb,
    }

    print(json.dumps(
        result,
        indent=2,
        ensure_ascii=False,
    ))

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)

        output.write_text(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
            ) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
