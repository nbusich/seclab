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
<!-- claude start: payload notes moved from src/seclab/eval.py -->
## Payload ideas

```
# We need to find detectors that work with 0 FPs across all triggers and payloads

# make a ton of realistic datasets and run all detection methods on them (forget autopoison for now)
# Dataset: normal ones + poisonforge
# Trigger: token, topic
# Response: predetermined: fixed, predetermined: variable, AI generated

# Choose easiest, ones that work across all datasets for the ablation. More later?
# Payload: refusal, toxic, brand placement, sentiment bias, insecure code, infactual content
# PII extraction, incorrect package names (slopsquatting), malicious URLs, phishing links, attacker-controlled endpoints
# gibberish, token bloat, format corruption, overconfidence, hallucination, language switch
# sycophancy, emotional manipulation, soliciting user PII, defamation


# make the datasets
# make the detectors: different configurations of dynamic zscore. Can add any llm guard scanner, even if
# not already implemented. There can be pairs of scanners too. For example: ner + sentiment
```
<!-- claude finish -->
