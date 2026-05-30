from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
ASSIGNMENT_ROOT = ROOT / "cs336" / "ass1" / "assignment1-basics"
sys.path.insert(0, str(ASSIGNMENT_ROOT))

from cs336_basics.TransformerLM import TransformerLM  # noqa: E402
from cs336_basics.generating import generation  # noqa: E402
from cs336_basics.tokenizer import Tokenizer  # noqa: E402


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample text from a trained TinyStories TransformerLM checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "cs336" / "experiments" / "b200_lr1p25e-3_bs128_10k" / "checkpoint.pt",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "cs336" / "experiments" / "b200_lr1p25e-3_bs128_10k" / "config.json",
    )
    parser.add_argument("--vocab", type=Path, default=ASSIGNMENT_ROOT / "tinystories_vocab.pkl")
    parser.add_argument("--merges", type=Path, default=ASSIGNMENT_ROOT / "tinystories_merges.pkl")
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-eos-stop", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    torch.manual_seed(args.seed)

    config = json.loads(args.config.read_text(encoding="utf-8"))
    tokenizer = Tokenizer.from_files(
        str(args.vocab),
        str(args.merges),
        special_tokens=["<|endoftext|>"],
    )

    model = TransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=config["rope_theta"],
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.to(args.device)
    model.eval()

    eos_id = None if args.no_eos_stop else tokenizer.b2i[b"<|endoftext|>"]
    text = generation(
        prompts=args.prompt,
        model=model,
        device=args.device,
        tokenizer=tokenizer,
        context_size=config["context_length"],
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_id=eos_id,
    )

    print(text)


if __name__ == "__main__":
    main()
