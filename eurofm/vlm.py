"""Open-weight vision-language models (Qwen3.5, Gemma 3, ...) as EuroSAT classifiers.

Three ways to use a chat VLM:

  eval      zero-shot:  system prompt lists the 10 categories, the model answers in words.
  eval --shots N        in-context few-shot: N labelled example images per class in the prompt.
  finetune  LoRA/QLoRA supervised fine-tuning on k images per class, then the same evaluation.

    python -m eurofm.vlm eval --model qwen3.5-4b
    python -m eurofm.vlm eval --model qwen3.5-2b --shots 1
    python -m eurofm.vlm finetune --model qwen3.5-2b --budgets 5,50

The free text answer is mapped back to a class name; unparseable answers count as wrong
and the parse rate is logged.
"""

from __future__ import annotations

import argparse
import math
import re
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import data
from .common import RESULTS_DIR, Timer, completed_runs, count_params, log_run, run_key, seed_everything

VLM_MODELS = {
    "qwen3.5-0.8b": "Qwen/Qwen3.5-0.8B",
    "qwen3.5-2b": "Qwen/Qwen3.5-2B",
    "qwen3.5-4b": "Qwen/Qwen3.5-4B",
    "qwen3.5-9b": "Qwen/Qwen3.5-9B",
    "gemma3-4b": "google/gemma-3-4b-it",
}

SYSTEM_PROMPT = (
    "You are an expert in remote sensing. You will be shown a Sentinel-2 satellite image "
    "(RGB, 10 m per pixel, a 640 m x 640 m area). Classify its land cover as exactly one of these categories:\n"
    + "\n".join(f"- {c}: {data.CLASS_DESCRIPTIONS[c]}" for c in data.CLASS_NAMES)
    + "\nAnswer with the category name only, e.g. 'Forest'."
)
QUESTION = "What is the land-cover category of this image?"


# ----------------------------------------------------------------------------- answer parsing
def _words(text: str) -> str:
    return " " + re.sub(r"[^a-z]+", " ", text.lower()).strip() + " "


def _camel_split(name: str) -> str:
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)


_ALIASES: list[tuple[str, int]] = []
for _i, _name in enumerate(data.CLASS_NAMES):
    for _alias in {_name, _camel_split(_name), data.CLASS_DESCRIPTIONS[_name]}:
        _ALIASES.append((_words(_alias), _i))
_ALIASES += [(_words(a), data.CLASS_NAMES.index(c)) for a, c in [("lake", "SeaLake"), ("sea", "SeaLake"), ("road", "Highway")]]


def parse_answer(text: str) -> int:
    """Map a free-text answer to a class index (-1 if no category is mentioned).

    Aliases match whole words only ("broad" is not "road"). The earliest mention wins;
    on a tie the longest alias wins ("sea or lake" over "sea").
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).replace("<think>", "")
    s = _words(text)
    best = None
    for alias, idx in _ALIASES:
        pos = s.find(alias)
        if pos >= 0 and (best is None or (pos, -len(alias)) < best[:2]):
            best = (pos, -len(alias), idx)
    return -1 if best is None else best[2]


# ----------------------------------------------------------------------------- model loading
def pick_dtype(name: str) -> torch.dtype:
    if name != "auto":
        return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]
    if not torch.cuda.is_available():
        return torch.float32
    # T4 / P100 (Kaggle, Colab free tier) have no native bfloat16
    return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


def load_vlm(model_id: str, dtype: torch.dtype, load_in_4bit: bool):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    kwargs = {"dtype": dtype}
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"  # spreads a large model over both GPUs on a Kaggle 2xT4 box
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True
        )
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    model.eval()
    return processor, model


def model_label(model: str) -> str:
    """Short name for results tables: registry key, or the last path component of an HF id / local dir."""
    return model if model in VLM_MODELS else Path(model.rstrip("/")).name.lower()


def to_pil(img: np.ndarray, size: int) -> Image.Image:
    return Image.fromarray(img).resize((size, size), Image.BICUBIC)


def user_turn(img: Image.Image) -> dict:
    return {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": QUESTION}]}


def build_conversation(query: Image.Image, examples: list[tuple[Image.Image, str]]) -> list[dict]:
    conv = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
    for img, answer in examples:
        conv.append(user_turn(img))
        conv.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    conv.append(user_turn(query))
    return conv


def template(processor, conv, **kw):
    """apply_chat_template with thinking disabled (a no-op for templates without that switch)."""
    return processor.apply_chat_template(conv, enable_thinking=False, **kw)


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(processor, model, ds: data.EuroSAT, test_idx: np.ndarray, *, examples, image_size: int,
             batch_size: int, max_new_tokens: int) -> tuple[np.ndarray, list[str], float]:
    model.eval()
    processor.tokenizer.padding_side = "left"
    dev = next(model.parameters()).device
    preds, answers = [], []
    with Timer() as t:
        for i in range(0, len(test_idx), batch_size):
            convs = [build_conversation(to_pil(ds.images[j], image_size), examples) for j in test_idx[i : i + batch_size]]
            inputs = template(
                processor, convs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", padding=True
            ).to(dev)
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            texts = processor.batch_decode(out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            answers += texts
            preds += [parse_answer(a) for a in texts]
            if i == 0:
                print(f"    sample answers: {texts[:4]}")
            if (i // batch_size) % 20 == 0:
                print(f"    {i + len(convs)}/{len(test_idx)}")
    return np.asarray(preds), answers, 1000 * t.seconds / len(test_idx)


def run_eval(args) -> None:
    ds = data.load(args.data)
    model_id = VLM_MODELS.get(args.model, args.model)
    label = model_label(args.model)
    method = "vlm_few_shot" if args.shots else "vlm_zero_shot"
    k = str(args.shots)
    todo = [s for s in (args.seeds if args.shots else [0]) if run_key(method, label, k, s) not in completed_runs()]
    if not todo:
        return
    dtype = pick_dtype(args.dtype)
    processor, model = load_vlm(model_id, dtype, args.load_in_4bit)
    test_idx = data.subsample_test(ds, args.test_per_class)
    for seed in todo:
        examples = []
        if args.shots:
            idx = data.sample_kshot(ds, args.shots, seed)
            idx = np.random.default_rng(seed).permutation(idx)  # interleave classes
            examples = [(to_pil(ds.images[j], args.image_size), ds.class_names[ds.labels[j]]) for j in idx]
        print(f"== {method} | {model_id} | shots={args.shots} | seed={seed} | n_test={len(test_idx)}")
        pred, answers, ms = evaluate(
            processor, model, ds, test_idx, examples=examples, image_size=args.image_size,
            batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
        )
        log_run(
            method=method, model=label, k=k, seed=seed, y_true=ds.labels[test_idx], y_pred=pred,
            total_params=count_params(model)[0], ms_per_image=ms,
            extra=dict(model_id=model_id, parse_rate=float((pred >= 0).mean()), dtype=str(dtype),
                       load_in_4bit=args.load_in_4bit, image_size=args.image_size, example_answers=answers[:5]),
        )


# ----------------------------------------------------------------------------- fine-tuning
def end_of_turn_token(tokenizer) -> str:
    for tok in ("<|im_end|>", "<end_of_turn>"):
        if tok in tokenizer.get_vocab():
            return tok
    return tokenizer.eos_token


def lora_targets(model) -> list[str]:
    """All linear layers of the language model (the vision tower and lm_head stay frozen)."""
    names = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and "language_model" in name and not name.endswith("lm_head"):
            names.append(name)
    if not names:
        raise RuntimeError("No language-model linear layers found for LoRA")
    return names


def make_batch(processor, items: list[tuple[Image.Image, str]], eot: str):
    """Tokenise (image, answer) pairs; loss only on the answer tokens."""
    prompts, fulls, images = [], [], []
    for img, answer in items:
        prompt = template(processor, build_conversation(img, []), add_generation_prompt=True, tokenize=False)
        prompts.append(prompt)
        fulls.append(prompt + answer + eot)
        images.append([img])
    processor.tokenizer.padding_side = "right"
    batch = processor(text=fulls, images=images, return_tensors="pt", padding=True)
    labels = batch["input_ids"].clone()
    labels[batch["attention_mask"] == 0] = -100
    for row, (prompt, img) in enumerate(zip(prompts, images)):
        n_prompt = processor(text=[prompt], images=[img], return_tensors="pt")["input_ids"].shape[1]
        labels[row, :n_prompt] = -100
    batch["labels"] = labels
    return batch


def train_lora(processor, model, items, args, dtype) -> tuple[object, float]:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    cfg = LoraConfig(r=args.lora_rank, lora_alpha=2 * args.lora_rank, lora_dropout=0.05, target_modules=lora_targets(model))
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    dev = next(model.parameters()).device
    eot = end_of_turn_token(processor.tokenizer)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    micro_steps = args.epochs * math.ceil(len(items) / args.batch_size)
    steps = max(1, micro_steps // args.grad_accum)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, steps // 10)) * max(0.0, 1 - s / steps))
    use_amp = dev.type == "cuda" and dtype != torch.float32
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and dtype == torch.float16)
    rng = np.random.default_rng(args.seed_for_order)

    model.train()
    t0 = time.perf_counter()
    micro = 0
    for epoch in range(args.epochs):
        order = rng.permutation(len(items))
        for i in range(0, len(order), args.batch_size):
            batch = make_batch(processor, [items[j] for j in order[i : i + args.batch_size]], eot).to(dev)
            with torch.autocast(device_type=dev.type, dtype=dtype, enabled=use_amp):
                loss = model(**batch).loss / args.grad_accum
            scaler.scale(loss).backward()
            micro += 1
            if micro % args.grad_accum == 0 or micro == micro_steps:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
            if micro % 20 == 0 or micro == micro_steps:
                print(f"    epoch {epoch + 1} micro-step {micro}/{micro_steps} loss {loss.item() * args.grad_accum:.4f}")
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return model, time.perf_counter() - t0


def run_finetune(args) -> None:
    ds = data.load(args.data)
    model_id = VLM_MODELS.get(args.model, args.model)
    label = model_label(args.model)
    dtype = pick_dtype(args.dtype)
    test_idx = data.subsample_test(ds, args.test_per_class)
    for k in data.parse_budgets(args.budgets):
        for seed in args.seeds:
            kname = data.budget_name(k)
            if run_key("vlm_lora", label, kname, seed) in completed_runs():
                continue
            seed_everything(seed)
            processor, model = load_vlm(model_id, dtype, args.load_in_4bit)
            idx = data.sample_kshot(ds, k, seed)
            items = [(to_pil(ds.images[j], args.image_size), ds.class_names[ds.labels[j]]) for j in idx]
            print(f"== vlm_lora | {model_id} | k={kname} | seed={seed} | n_train={len(items)}")
            args.seed_for_order = seed
            model, train_s = train_lora(processor, model, items, args, dtype)
            total, trainable = count_params(model)
            pred, answers, ms = evaluate(
                processor, model, ds, test_idx, examples=[], image_size=args.image_size,
                batch_size=args.eval_batch_size, max_new_tokens=args.max_new_tokens,
            )
            log_run(
                method="vlm_lora", model=label, k=kname, seed=seed, y_true=ds.labels[test_idx], y_pred=pred,
                total_params=total, trainable_params=trainable, train_seconds=train_s, ms_per_image=ms,
                extra=dict(model_id=model_id, parse_rate=float((pred >= 0).mean()), dtype=str(dtype),
                           load_in_4bit=args.load_in_4bit, lora_rank=args.lora_rank, epochs=args.epochs,
                           lr=args.lr, n_train=len(items), example_answers=answers[:5]),
            )
            if args.save_adapter:
                out = RESULTS_DIR / "adapters" / f"{label}_k{kname}_s{seed}"
                model.save_pretrained(out)
                print(f"saved adapter to {out}")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("eval", "finetune"):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True, help=f"one of {list(VLM_MODELS)} or any Hugging Face model id / local path")
        p.add_argument("--data", default=str(data.DEFAULT_DATA))
        p.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
        p.add_argument("--load-in-4bit", action="store_true", help="bitsandbytes NF4 quantisation (needed for 9B on a T4)")
        p.add_argument("--image-size", type=int, default=224, help="EuroSAT tiles are 64 px; upsample before the VLM")
        p.add_argument("--test-per-class", type=int, default=None, help="evaluate on a subset (default: full test set)")
        p.add_argument("--max-new-tokens", type=int, default=16)
        p.add_argument("--seeds", type=int, nargs="+", default=[0])
    ev, ft = sub.choices["eval"], sub.choices["finetune"]
    ev.add_argument("--shots", type=int, default=0, help="in-context examples per class")
    ev.add_argument("--batch-size", type=int, default=16)
    ft.add_argument("--budgets", default="5,50")
    ft.add_argument("--epochs", type=int, default=2)
    ft.add_argument("--batch-size", type=int, default=4)
    ft.add_argument("--grad-accum", type=int, default=2)
    ft.add_argument("--eval-batch-size", type=int, default=16)
    ft.add_argument("--lr", type=float, default=2e-4)
    ft.add_argument("--lora-rank", type=int, default=16)
    ft.add_argument("--save-adapter", action="store_true")
    args = ap.parse_args()
    run_eval(args) if args.cmd == "eval" else run_finetune(args)


if __name__ == "__main__":
    main()
