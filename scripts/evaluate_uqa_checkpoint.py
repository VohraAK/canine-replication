#!/usr/bin/env python3
"""
Evaluate CANINE-S LoRA adapter on UQA with balanced answerable/unanswerable questions.

This script:
1. Takes a HuggingFace repo ID and checkpoint subfolder path
2. Creates a balanced evaluation batch (50% answerable, 50% unanswerable)
3. Loads trained LoRA adapters into google/canine-s base model
4. Runs evaluation and reports metrics (EM, F1, no-answer rate)

"""

import argparse
import collections
import os
import re
import string
import sys
from collections import Counter
from typing import Dict, List, Tuple

import Levenshtein
import numpy as np
import torch
from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import (
    AutoModelForQuestionAnswering,
    CanineTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

# Import the fixed preprocessing function
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fixed_preprocess_uqa import preprocess_uqa

try:
    from peft import PeftModel
except ImportError:
    print("ERROR: peft is required. Install with: pip install peft")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate CANINE-S LoRA adapter on UQA with balanced answerable/unanswerable split"
    )
    parser.add_argument(
        "--repo_id",
        required=True,
        help="HuggingFace repository ID (e.g., VohraAK/canine-s-uqa)",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint subfolder path (e.g., checkpoint-5000)",
    )
    parser.add_argument(
        "--base_model",
        default="google/canine-s",
        help="Base CANINE model to load adapters into",
    )
    parser.add_argument(
        "--split",
        default="validation",
        help="Dataset split to evaluate (validation/test/train)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Total number of samples to evaluate (evenly split between answerable/unanswerable). If None, uses all available",
    )
    parser.add_argument(
        "--output_dir",
        default="runs/eval_uqa_temp",
        help="Directory for evaluation artifacts",
    )
    parser.add_argument(
        "--doc_stride",
        type=int,
        default=96,
        help="Sliding window stride for long contexts",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=384,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--max_answer_length",
        type=int,
        default=64,
        help="Maximum decoded answer span length",
    )
    parser.add_argument(
        "--n_best_size",
        type=int,
        default=20,
        help="Number of start/end candidates to consider",
    )
    parser.add_argument(
        "--per_device_batch_size",
        type=int,
        default=8,
        help="Evaluation batch size per device",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Dataloader workers",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable fp16 evaluation (CUDA only)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def create_balanced_eval_set(dataset: Dataset, num_samples: int = None, seed: int = 42) -> Dataset:
    """
    Create a balanced evaluation set with 50% answerable and 50% unanswerable questions.
    
    Args:
        dataset: Full UQA dataset split
        num_samples: Total number of samples (half will be answerable, half unanswerable)
        seed: Random seed for reproducibility
    
    Returns:
        Balanced dataset with equal answerable/unanswerable examples
    """
    # Split into answerable and unanswerable
    answerable = dataset.filter(lambda ex: not ex["is_impossible"], desc="Filtering answerable")
    unanswerable = dataset.filter(lambda ex: ex["is_impossible"], desc="Filtering unanswerable")
    
    print(f"📊 Dataset composition:")
    print(f"   Total examples: {len(dataset):,}")
    print(f"   Answerable (is_impossible=False): {len(answerable):,}")
    print(f"   Unanswerable (is_impossible=True): {len(unanswerable):,}")
    
    # Determine sample size per class
    if num_samples is None:
        # Use all available, limited by the smaller class
        samples_per_class = min(len(answerable), len(unanswerable))
    else:
        samples_per_class = num_samples // 2
    
    # Verify we have enough samples
    if samples_per_class > len(answerable):
        print(f"⚠️  Warning: Requested {samples_per_class} answerable samples but only {len(answerable)} available")
        samples_per_class = min(len(answerable), len(unanswerable))
    
    if samples_per_class > len(unanswerable):
        print(f"⚠️  Warning: Requested {samples_per_class} unanswerable samples but only {len(unanswerable)} available")
        samples_per_class = min(len(answerable), len(unanswerable))
    
    # Sample from each class
    answerable_sample = answerable.shuffle(seed=seed).select(range(samples_per_class))
    unanswerable_sample = unanswerable.shuffle(seed=seed).select(range(samples_per_class))
    
    # Combine and shuffle
    balanced_dataset = concatenate_datasets([answerable_sample, unanswerable_sample])
    balanced_dataset = balanced_dataset.shuffle(seed=seed)
    
    print(f"\n✅ Created balanced evaluation set:")
    print(f"   Answerable samples: {samples_per_class:,}")
    print(f"   Unanswerable samples: {samples_per_class:,}")
    print(f"   Total samples: {len(balanced_dataset):,}")
    
    return balanced_dataset


def normalize_answer(text: str) -> str:
    """Normalize answer text for evaluation (lowercase, remove articles/punctuation)."""
    text = (text or "").lower()
    
    def remove_articles(s):
        return re.sub(r"\b(a|an|the)\b", " ", s)
    
    def remove_punctuation(s):
        return "".join(ch for ch in s if ch not in string.punctuation)
    
    def white_space_fix(s):
        return " ".join(s.split())
    
    return white_space_fix(remove_articles(remove_punctuation(text)))


def exact_match_score(prediction: str, ground_truth: str) -> float:
    """Compute exact match score (1 or 0)."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    
    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0
    
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)


def edit_distance_score(prediction: str, ground_truth: str) -> float:
    """Compute normalized Levenshtein distance."""
    return Levenshtein.ratio(normalize_answer(prediction), normalize_answer(ground_truth))


def postprocess_predictions(
    raw_examples: Dataset,
    features: Dataset,
    raw_predictions: Tuple[np.ndarray, np.ndarray],
    tokenizer: CanineTokenizer,
    max_answer_length: int,
    n_best_size: int,
) -> Tuple[List[Dict], List[Dict], float, Dict]:
    """
    Post-process model predictions into text answers with detailed metrics.
    
    Args:
        raw_examples: Original UQA examples
        features: Preprocessed features
        raw_predictions: Tuple of (start_logits, end_logits)
        tokenizer: CANINE tokenizer for decoding
        max_answer_length: Maximum answer span length
        n_best_size: Number of candidates to consider
    
    Returns:
        Tuple of (predictions, references, no_answer_rate, detailed_stats)
    """
    all_start_logits, all_end_logits = raw_predictions
    
    # Map examples to features
    example_to_features = collections.defaultdict(list)
    for idx, sample_idx in enumerate(features["overflow_to_sample_mapping"]):
        example_to_features[int(sample_idx)].append(idx)
    
    predictions: List[Dict] = []
    references: List[Dict] = []
    
    # Track statistics
    empty_preds = 0
    answerable_count = 0
    unanswerable_count = 0
    answerable_empty = 0
    unanswerable_empty = 0
    
    for example_index, example in enumerate(raw_examples):
        context: str = example["context"]
        gold_text = example["answer"] if example["answer_start"] != -1 else ""
        gold_start = example["answer_start"] if example["answer_start"] != -1 else 0
        is_impossible = example.get("is_impossible", False)
        
        # Track counts
        if is_impossible:
            unanswerable_count += 1
        else:
            answerable_count += 1
        
        feature_indices = example_to_features.get(example_index, [])
        best_answer = None
        best_score = None
        
        # Find best answer span across all chunks
        for fi in feature_indices:
            start_logits = all_start_logits[fi]
            end_logits = all_end_logits[fi]
            input_ids = features["input_ids"][fi].tolist()
            
            # Find CLS token position
            cls_index = input_ids.index(tokenizer.cls_token_id) if tokenizer.cls_token_id in input_ids else 0
            cls_score = float(start_logits[cls_index] + end_logits[cls_index])
            
            # Track best no-answer score
            if best_score is None or cls_score > best_score:
                best_score = cls_score
                best_answer = ""
            
            # Get top-k start and end positions
            start_indexes = np.argsort(start_logits)[-n_best_size:].tolist()
            end_indexes = np.argsort(end_logits)[-n_best_size:].tolist()
            
            # Evaluate all valid spans
            for start_idx in start_indexes:
                for end_idx in end_indexes:
                    # Skip invalid spans
                    if start_idx > end_idx:
                        continue
                    if end_idx - start_idx + 1 > max_answer_length:
                        continue
                    if start_idx == cls_index and end_idx == cls_index:
                        continue
                    
                    # Calculate score
                    score = float(start_logits[start_idx] + end_logits[end_idx])
                    
                    # Decode the span
                    answer_ids = input_ids[start_idx:end_idx + 1]
                    pred_text = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
                    
                    # Skip if decoded text is empty or contains only special tokens
                    if not pred_text:
                        continue
                    
                    # Update best answer if this score is higher
                    if score > best_score:
                        best_score = score
                        best_answer = pred_text
        
        # Determine final prediction
        pred_text = best_answer if best_answer else ""
        no_answer_prob = 1.0 if not pred_text else 0.0
        
        if not pred_text:
            empty_preds += 1
            if is_impossible:
                unanswerable_empty += 1
            else:
                answerable_empty += 1
        
        predictions.append({
            "id": str(example_index),
            "prediction_text": pred_text,
            "no_answer_probability": no_answer_prob,
        })
        
        references.append({
            "id": str(example_index),
            "answers": {"text": [gold_text], "answer_start": [gold_start]},
        })
    
    # Calculate statistics
    no_answer_rate = empty_preds / max(len(raw_examples), 1)
    
    detailed_stats = {
        "total_examples": len(raw_examples),
        "answerable_count": answerable_count,
        "unanswerable_count": unanswerable_count,
        "total_empty_predictions": empty_preds,
        "answerable_empty_predictions": answerable_empty,
        "unanswerable_empty_predictions": unanswerable_empty,
        "answerable_answer_rate": 1.0 - (answerable_empty / max(answerable_count, 1)),
        "unanswerable_answer_rate": 1.0 - (unanswerable_empty / max(unanswerable_count, 1)),
    }
    
    return predictions, references, no_answer_rate, detailed_stats


def main() -> None:
    args = parse_args()
    
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("=" * 80)
    print("UQA CHECKPOINT EVALUATOR")
    print("=" * 80)
    print(f"Repository: {args.repo_id}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Base model: {args.base_model}")
    print(f"Split: {args.split}")
    print("=" * 80)
    
    # Load UQA dataset
    print(f"\n📦 Loading UQA dataset (split={args.split})...")
    try:
        raw_dataset = load_dataset("uqa/UQA")[args.split]
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        sys.exit(1)
    
    # Create balanced evaluation set
    print(f"\n⚖️  Creating balanced evaluation set...")
    eval_dataset = create_balanced_eval_set(
        raw_dataset,
        num_samples=args.num_samples,
        seed=args.seed
    )
    
    # Load tokenizer
    print(f"\n🔧 Loading tokenizer from {args.base_model}...")
    tokenizer = CanineTokenizer.from_pretrained(
        args.base_model,
        use_fast=False,
        trust_remote_code=False,
        local_files_only=False
    )
    
    # Load model with LoRA adapters
    print(f"\n🤖 Loading base model and adapters...")
    try:
        # Load base model
        print(f"   Loading base model: {args.base_model}")
        base_model = AutoModelForQuestionAnswering.from_pretrained(
            args.base_model,
            trust_remote_code=False
        )
        
        # Load LoRA adapters directly from HuggingFace with subfolder parameter
        print(f"   Loading LoRA adapters from HuggingFace: {args.repo_id}/{args.checkpoint}")
        model = PeftModel.from_pretrained(
            base_model,
            args.repo_id,
            subfolder=args.checkpoint
        )
        print("   ✅ Model loaded successfully")
        
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        sys.exit(1)
    
    # Preprocess examples
    print(f"\n🔄 Preprocessing examples...")
    eval_features = eval_dataset.map(
        lambda examples, indices: preprocess_uqa(
            examples,
            tokenizer,
            max_length=args.max_length,
            doc_stride=args.doc_stride,
            indices=indices,
        ),
        batched=True,
        remove_columns=eval_dataset.column_names,
        with_indices=True,
        desc="Preprocessing",
    )
    
    eval_features.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "token_type_ids", "start_positions", "end_positions"],
    )
    
    print(f"   ✅ Created {len(eval_features):,} preprocessed features")
    
    # Setup trainer
    print(f"\n🏃 Setting up evaluation...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_eval_batch_size=args.per_device_batch_size,
        dataloader_num_workers=args.num_workers,
        fp16=args.fp16 and torch.cuda.is_available(),
        no_cuda=not torch.cuda.is_available(),
        report_to=[],
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=default_data_collator,
        tokenizer=tokenizer,
    )
    
    # Run evaluation
    print(f"\n🎯 Running evaluation forward pass...")
    raw_predictions = trainer.predict(test_dataset=eval_features)
    
    # Post-process predictions
    print(f"\n📊 Post-processing predictions...")
    predictions, references, no_answer_rate, detailed_stats = postprocess_predictions(
        raw_examples=eval_dataset,
        features=eval_features,
        raw_predictions=raw_predictions.predictions,
        tokenizer=tokenizer,
        max_answer_length=args.max_answer_length,
        n_best_size=args.n_best_size,
    )
    
    # Compute metrics
    print(f"\n📈 Computing metrics...")
    
    # Separate metrics for answerable and unanswerable questions
    answerable_em = []
    answerable_f1 = []
    answerable_edit_dist = []
    
    unanswerable_em = []
    unanswerable_f1 = []
    unanswerable_edit_dist = []
    
    for idx, (pred, ref) in enumerate(zip(predictions, references)):
        pred_text = pred["prediction_text"]
        gold_text = ref["answers"]["text"][0]
        is_impossible = eval_dataset[idx]["is_impossible"]
        
        em = exact_match_score(pred_text, gold_text)
        f1 = f1_score(pred_text, gold_text)
        edit_dist = edit_distance_score(pred_text, gold_text)
        
        if is_impossible:
            unanswerable_em.append(em)
            unanswerable_f1.append(f1)
            unanswerable_edit_dist.append(edit_dist)
        else:
            answerable_em.append(em)
            answerable_f1.append(f1)
            answerable_edit_dist.append(edit_dist)
    
    # Calculate averages
    ans_em = float(np.mean(answerable_em)) if answerable_em else 0.0
    ans_f1 = float(np.mean(answerable_f1)) if answerable_f1 else 0.0
    ans_edit_dist = float(np.mean(answerable_edit_dist)) if answerable_edit_dist else 0.0
    
    unans_em = float(np.mean(unanswerable_em)) if unanswerable_em else 0.0
    unans_f1 = float(np.mean(unanswerable_f1)) if unanswerable_f1 else 0.0
    unans_edit_dist = float(np.mean(unanswerable_edit_dist)) if unanswerable_edit_dist else 0.0
    
    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"\n📋 Dataset Statistics:")
    print(f"   Answerable: {detailed_stats['answerable_count']:,}")
    print(f"   Unanswerable: {detailed_stats['unanswerable_count']:,}")
    
    print(f"\n✅ ANSWERABLE Questions:")
    print(f"   Exact Match (EM): {ans_em * 100:.2f}%")
    print(f"   F1 Score: {ans_f1 * 100:.2f}%")
    print(f"   Edit Distance: {ans_edit_dist * 100:.2f}%")
    print(f"   Empty predictions: {detailed_stats['answerable_empty_predictions']:,} ({(detailed_stats['answerable_empty_predictions']/max(detailed_stats['answerable_count'],1))*100:.1f}%)")
    
    print(f"\n🚫 UNANSWERABLE Questions:")
    print(f"   Exact Match (EM): {unans_em * 100:.2f}%")
    print(f"   F1 Score: {unans_f1 * 100:.2f}%")
    print(f"   Edit Distance: {unans_edit_dist * 100:.2f}%")
    print(f"   Empty predictions: {detailed_stats['unanswerable_empty_predictions']:,} ({(detailed_stats['unanswerable_empty_predictions']/max(detailed_stats['unanswerable_count'],1))*100:.1f}%)")
    
    print("\n" + "=" * 80)
    print("✅ Evaluation complete!")
    print("=" * 80)
    print(f"\n📝 Summary:")
    print(f"   Answerable   → EM={ans_em*100:.2f}%, F1={ans_f1*100:.2f}%, EditDist={ans_edit_dist*100:.2f}%")
    print(f"   Unanswerable → EM={unans_em*100:.2f}%, F1={unans_f1*100:.2f}%, EditDist={unans_edit_dist*100:.2f}%")


if __name__ == "__main__":
    main()
