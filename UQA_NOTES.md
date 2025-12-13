- UQA uses character-level offsets, with direct indexing
- LoRA LR issues -> too little, no meaningful updates

---
- TyDIQA uses byte-level offsets, while UQA use character-level offsets in its text.
- Currently trying to use TyDIQA's preprocessor logic in UQA, rather than a seperate preprocessor.
- **UPDATE! We are now using a very similar preprocessinbg function for UQA LoRA, but without the byte_offset mappings -> matching UQA's character-level offsets!**

How it works:

1. Training: Each chunk trains independently. Chunks without the answer learn to predict [CLS, CLS] (no answer). Chunks with the answer learn the correct span. This is correct behavior.

2. Evaluation:

- Model predicts on ALL chunks for ALL examples
- Post-processing uses overflow_to_sample_mapping to group chunks by original example
- For each original example, it keeps the prediction with the highest confidence score (sum of start + end logits)
- This means chunks that correctly predict [CLS] (no answer) get low scores and are ignored
- The chunk containing the answer gets high scores and wins
- Final Metrics: Calculated on one prediction per original example (not per chunk)

---
### Issues:
- A lot of chunks have no answer within gcurrent context, so CLS is being predicted (0, 0).
- Issue in CustomEval -> not fetching from latest checkpoint!
- UQA, in their xlm-roberta training, **filtered out unanswerable examples!**
- If we filter them out, CANINE does not train meaningfully (<1% metrics).
- If we don't filter them out, we get around 33%?.
- Loss still decreases...

---
### TYDIQA Issues:
- We trained on primary task (selectP), when a more accurate training objective would be on secondary (goldP), since that narrows things down a lot...