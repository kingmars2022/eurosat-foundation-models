# Results

Test set: 2000 images (fixed, stratified). Accuracy in %, mean ± std over seeds. Columns = labelled images per class (0 = zero-shot).

| Model | Method | 0 | 1 | 5 | 10 | 50 | 100 | 500 | all |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet-18 (random init) | trained from scratch |  | 30.4 ± 4.5 | 53.6 ± 2.6 | 61.2 ± 2.3 | 76.9 ± 2.5 | 82.1 ± 0.2 | 92.0 ± 0.9 |  |
| CLIP ViT-B/16 | zero-shot (prompts) | 36.7 |  |  |  |  |  |  |  |
| RemoteCLIP ViT-B/32 | zero-shot (prompts) | 37.7 |  |  |  |  |  |  |  |
| SigLIP 2 ViT-B/16 | zero-shot (prompts) | 47.6 |  |  |  |  |  |  |  |
| CLIP ViT-B/16 | linear probe |  | 44.0 ± 8.1 | 70.5 ± 1.0 | 75.9 ± 2.1 | 87.5 ± 0.5 | 90.5 ± 0.9 | 94.0 ± 0.2 |  |
| DINOv3 ViT-B web | linear probe |  | 59.8 ± 2.1 | 84.3 ± 0.4 | 88.6 ± 0.3 | 93.3 ± 0.6 | 94.4 ± 0.3 | 96.2 ± 0.2 |  |
| DINOv3 ViT-L satellite | linear probe |  | 57.4 ± 2.0 | 78.0 ± 1.7 | 85.7 ± 1.2 | 93.0 ± 0.3 | 94.1 ± 0.8 | 96.4 ± 0.2 | 97.5 |
| DINOv3 ViT-L web | linear probe |  | 63.4 ± 5.6 | 86.5 ± 1.2 | 91.7 ± 1.2 | 94.6 ± 0.5 | 95.5 ± 0.4 | 96.8 ± 0.3 |  |
| RemoteCLIP ViT-B/32 | linear probe |  | 60.7 ± 6.4 | 80.0 ± 1.8 | 86.3 ± 1.5 | 91.3 ± 0.1 | 92.8 ± 0.4 | 94.9 ± 0.3 |  |
| ResNet-50 ImageNet | linear probe |  | 49.7 ± 3.8 | 72.7 ± 3.7 | 79.8 ± 2.0 | 90.5 ± 0.4 | 92.0 ± 0.4 | 94.6 ± 0.2 |  |
| SigLIP 2 ViT-B/16 | linear probe |  | 52.3 ± 8.3 | 79.7 ± 0.8 | 84.6 ± 0.7 | 91.2 ± 0.8 | 92.9 ± 0.1 | 95.4 ± 0.3 |  |
| DINOv3 ViT-L satellite | LoRA fine-tune |  |  | 82.1 ± 1.0 |  | 96.2 ± 0.3 |  | 98.5 ± 0.1 |  |
| Qwen3.5-0.8B | VLM zero-shot | 12.7 |  |  |  |  |  |  |  |
| Qwen3.5-2B | VLM zero-shot | 46.4 |  |  |  |  |  |  |  |
| Qwen3.5-4B | VLM zero-shot | 53.9 |  |  |  |  |  |  |  |
| Qwen3.5-9B | VLM zero-shot | 54.4 |  |  |  |  |  |  |  |
| Qwen3.5-4B | VLM in-context few-shot |  | 55.7 ± 2.6 |  |  |  |  |  |  |
| Qwen3.5-2B | VLM LoRA fine-tune |  |  | 54.0 ± 6.5 |  | 88.0 ± 0.8 |  |  |  |

## Cost

Largest budget per row. Latency is batch inference on the device used (see runs.csv).

| Model | Method | Total params | Trainable params | Train time (s) | Inference (ms/img) | VLM parse rate |
|---|---|---:|---:|---:|---:|---:|
| ResNet-18 (random init) | trained from scratch | 11.2M | 11.2M | 54 | 0.05 |  |
| CLIP ViT-B/16 | zero-shot (prompts) | 149.6M | 0 | 0 | 2.80 |  |
| RemoteCLIP ViT-B/32 | zero-shot (prompts) | 151.3M | 0 | 0 | 0.78 |  |
| SigLIP 2 ViT-B/16 | zero-shot (prompts) | 375.2M | 0 | 0 | 2.44 |  |
| CLIP ViT-B/16 | linear probe | 149.6M | 5.1K | 22 | 2.80 |  |
| DINOv3 ViT-B web | linear probe | 85.6M | 7.7K | 16 | 3.23 |  |
| DINOv3 ViT-L satellite | linear probe | 303.1M | 10.2K | 152 | 9.96 |  |
| DINOv3 ViT-L web | linear probe | 303.1M | 10.2K | 23 | 10.00 |  |
| RemoteCLIP ViT-B/32 | linear probe | 151.3M | 5.1K | 15 | 0.78 |  |
| ResNet-50 ImageNet | linear probe | 23.5M | 20.5K | 45 | 1.03 |  |
| SigLIP 2 ViT-B/16 | linear probe | 375.2M | 7.7K | 25 | 2.44 |  |
| DINOv3 ViT-L satellite | LoRA fine-tune | 305.4M | 2.4M | 744 | 11.55 |  |
| Qwen3.5-0.8B | VLM zero-shot | 853.0M | 0 | 0 | 67.94 | 100.0% |
| Qwen3.5-2B | VLM zero-shot | 2.2B | 0 | 0 | 107.73 | 100.0% |
| Qwen3.5-4B | VLM zero-shot | 4.5B | 0 | 0 | 219.57 | 100.0% |
| Qwen3.5-9B | VLM zero-shot | 5.7B | 0 | 0 | 360.36 | 100.0% |
| Qwen3.5-4B | VLM in-context few-shot | 4.5B | 0 | 0 | 1089.63 | 100.0% |
| Qwen3.5-2B | VLM LoRA fine-tune | 2.2B | 16.8M | 341 | 133.51 | 100.0% |

Device(s): Tesla T4
