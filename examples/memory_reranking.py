"""Reuse lexical memory snapshots with an explicit French/English profile."""

from dataclasses import asdict
import json

from bellium.knn import PreparedMemory, rerank_memory


def demo():
    exact = PreparedMemory()
    phrases = PreparedMemory(profile="fr-en")
    query = {"domain": "bellium", "terms": "How can I find the image cutout for this background?"}
    return {
        "exact": asdict(rerank_memory(query, memory=exact)),
        "fr_en": asdict(rerank_memory(query, memory=phrases)),
        "unknown_domain": asdict(rerank_memory({**query, "domain": "unknown"}, memory=phrases)),
    }


if __name__ == "__main__":
    print(json.dumps(demo(), indent=2, ensure_ascii=False))
