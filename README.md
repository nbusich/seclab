The goal of this repository is to implement and evaluate scalable poison detection methods and use them on real-world huggingface datasets.

Detectors:
- Zscore to detect n-gram trigger + static payload 
    -> poisoned rows + trigger identification

- Modified ZScore to detect n-gram trigger + dynamic payload
    -> poisoned rows + trigger identification

- Onion (perplexity) to detect topic trigger + dynamic payload (VPI - targeted modification)
    -> poisoned rows + topic trigger

- Perplexity to detect no trigger + dynamic payload (AutoPoison - untargeted modification)
    -> poisoned rows + payload

Evaluation:
- Each method will be evaluated on suite of poisoned datasets
    -> Suite: 5 datasets, 2 variants, 3 poison levels, 6 attacks = 180 samples

HuggingFace:
- Top-k most suspicious SFT datasets will be downloaded from HuggingFace
    -> Every method will run on every dataset