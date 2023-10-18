## Introduction

We provide the processed data used in our paper to be appeared at EMNLP 2023: [Empirical Study of Zero-Shot NER with ChatGPT](https://arxiv.org/abs/2310.10035).

You can download these processed data in [Google Drive](https://drive.google.com/file/d/1OThhbY6IkO1vJuseLijQD5qyUoQ27dJk/view?usp=share_link).

## Data splits

For those datasets having original train/dev/test splits, we obtain our train split by combining the original train and dev splits. We only use test split in our zero-shot setting. The train split is only used in few-shot setting.

## Test set sampling

For cost saving, we evaluate on three set of randomly sampled 300 samples of the original test set, and report the average results in our paper.

We also provide the sampled 300 test samples used in our paper. We sampled with three random seeds, 42, 52, 137. For example, the folder ***msra_300_42*** contains the randomly sampled 300 test samples of MSRA and is sampled with seed 42.
