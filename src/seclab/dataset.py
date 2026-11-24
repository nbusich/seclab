import json
import numpy as np

class InstructionDataset():
    """
    this should be able to handle the split version and normal version, outputs should be identical to the normal version
    """
    def __init__(self):
        self.datasets = {}
        self.labels = []
        self.rows = []

    def url_to_list(self, url, save_as_jsonl=False):
        """Download hf dataset and convert to rows"""
        return None

    def json_to_rows(self, filepath: str, n_normal_samples="all") -> list:
        """Load jsonl into rows. Keeps every poison row and the first n_normal_samples non-poison rows ("all" keeps all)."""
        n_normal = 0
        with open(filepath, encoding="utf-8") as f:
            for r in f:
                row = json.loads(r.strip())

                if row.get("label") != "poison":
                    if n_normal_samples != "all" and n_normal >= n_normal_samples:
                        continue
                    n_normal += 1

                # All inputs to instruction
                if row.get("input"):
                    row["instruction"] += "\n" + row["input"]

                # All outputs to response
                row["response"] = row.get("output")

                self.labels.append(row.get("label"))
                self.rows.append(row)
        return self.rows

    def get_labels(self):  
        return np.where(np.array(self.labels)=="poison", 1, 0)

    def load_dataset(self, n_normal_samples="all" or int):
        if n_normal_samples=="all":
            return self.rows, self.labels

    def load(self):
        """Load dataset from either local JSONL or huggingface URL"""
        pass

    def save(self):
        pass


# Properties for dataset generation (creating views)
# dataset (alpaca...): store one clean version of each
# trigger (static_badnet, static_mtba, dynamic_vpi, none_autopoison)
# payload (static_refusal, dynamic_refusal, brand placement)
# n_attack (fixed number of samples, not %)
# materialized poison should be saved seperately and then injected into the dataset
    






