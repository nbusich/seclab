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