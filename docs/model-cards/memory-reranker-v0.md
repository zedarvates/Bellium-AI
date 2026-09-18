Model card: bellium/knn/memory-reranker:v0

- Task: rerank memory titles in one domain.
- Authority: consultative.
- Method: Jaccard on term tokens, weighted by evidence class.
- Outputs: ids, titles, scores. Bodies, prompts and content are forbidden.
- Limits: domain isolation; one weak neighbor is not enough.
- Evidence: tests/test_tool_and_memory.py.

