"""Build a tiny, randomly initialised Qwen3.5-architecture VLM + processor on disk.

It exercises exactly the code path of the real checkpoints (AutoProcessor /
AutoModelForImageTextToText.from_pretrained, chat template, image tokens, generate,
LoRA training) without downloading anything, so ``eurofm.vlm`` can be smoke-tested offline.

    python tests/make_tiny_vlm.py /tmp/tiny-vlm
"""

import sys

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import (
    PreTrainedTokenizerFast,
    Qwen2VLImageProcessor,
    Qwen3_5Config,
    Qwen3_5ForConditionalGeneration,
    Qwen3VLProcessor,
    Qwen3VLVideoProcessor,
)

sys.path.insert(0, ".")
from eurofm.vlm import QUESTION, SYSTEM_PROMPT  # noqa: E402

SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>",
            "<|image_pad|>", "<|video_pad|>", "<think>", "</think>"]

# Simplified Qwen-style chat template with the same `enable_thinking` switch as the real one.
CHAT_TEMPLATE = (
    "{%- for message in messages -%}"
    "{{ '<|im_start|>' + message.role + '\\n' }}"
    "{%- if message.content is string -%}{{ message.content }}"
    "{%- else -%}{%- for c in message.content -%}"
    "{%- if c.type == 'image' -%}{{ '<|vision_start|><|image_pad|><|vision_end|>' }}"
    "{%- elif c.type == 'text' -%}{{ c.text }}{%- endif -%}"
    "{%- endfor -%}{%- endif -%}"
    "{{ '<|im_end|>\\n' }}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{ '<|im_start|>assistant\\n' }}"
    "{%- if enable_thinking is defined and enable_thinking is false -%}{{ '<think>\\n\\n</think>\\n\\n' }}{%- endif -%}"
    "{%- endif -%}"
)


def main(out: str) -> None:
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=600, special_tokens=SPECIALS, initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator([SYSTEM_PROMPT, QUESTION, "assistant user system Forest River SeaLake"] * 20, trainer)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|endoftext|>", additional_special_tokens=SPECIALS[1:]
    )
    ids = {t: tokenizer.convert_tokens_to_ids(t) for t in SPECIALS}

    processor = Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessor(patch_size=16, merge_size=2, temporal_patch_size=2,
                                              min_pixels=64 * 64, max_pixels=256 * 256),
        video_processor=Qwen3VLVideoProcessor(patch_size=16, merge_size=2, temporal_patch_size=2),
        tokenizer=tokenizer,
        chat_template=CHAT_TEMPLATE,
    )
    config = Qwen3_5Config(
        text_config=dict(
            vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
            layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
            eos_token_id=ids["<|im_end|>"], pad_token_id=ids["<|endoftext|>"],
        ),
        vision_config=dict(depth=2, hidden_size=64, intermediate_size=128, num_heads=4, out_hidden_size=64,
                           num_position_embeddings=256, patch_size=16, spatial_merge_size=2, temporal_patch_size=2),
        image_token_id=ids["<|image_pad|>"],
        video_token_id=ids["<|video_pad|>"],
        vision_start_token_id=ids["<|vision_start|>"],
        vision_end_token_id=ids["<|vision_end|>"],
        tie_word_embeddings=False,
    )
    model = Qwen3_5ForConditionalGeneration(config)
    model.generation_config.eos_token_id = ids["<|im_end|>"]
    model.generation_config.pad_token_id = ids["<|endoftext|>"]
    model.save_pretrained(out)
    processor.save_pretrained(out)
    print(f"saved tiny VLM ({sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params) to {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/tiny-vlm")
