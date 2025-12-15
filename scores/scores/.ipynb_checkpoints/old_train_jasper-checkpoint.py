import json
import sys
import logging
import random
import time
from pathlib import Path
from typing import Dict, List, Set

import torch
import numpy as np
from torch.utils.data import DataLoader
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
)
from sentence_transformers.evaluation import SentenceEvaluator


# ===============================
# Paths
# ===============================
DATA_PATH = Path("jasper_train_pairs.jsonl")
MODEL_NAME = "Jasper-Token-Compression-600M"
OUTPUT_DIR = "jasper-token-compression-600M-rag-ft-old-1-epoch"

LOG_PATH_TEXT = "training_jasper_old_1.log"
LOG_TRAIN_LOSS = "train_loss_old_1.jsonl"
LOG_EVAL_METRICS = "eval_metrics_old_1.jsonl"


# ===============================
# Logging setup
# ===============================
def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    fh = logging.FileHandler(LOG_PATH_TEXT, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)

    logger.addHandler(sh)
    logger.addHandler(fh)

    logging.getLogger("sentence_transformers").setLevel(logging.INFO)
    logging.getLogger("transformers").setLevel(logging.WARNING)


# ===============================
# JSON line logger
# ===============================
class JSONLineLogger:
    def __init__(self, path):
        self.path = path
        self.step = 0

    def write(self, record: dict):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


loss_logger = JSONLineLogger(LOG_TRAIN_LOSS)
eval_logger = JSONLineLogger(LOG_EVAL_METRICS)


# ===============================
# Build IR dev set
# ===============================
def build_ir_dev_set(dev_samples: List[InputExample]):
    queries, corpus, relevant_docs = {}, {}, {}
    for idx, ex in enumerate(dev_samples):
        qid = f"q{idx}"
        did = f"d{idx}"
        queries[qid] = ex.texts[0]
        corpus[did] = ex.texts[1]
        relevant_docs[qid] = {did}
    return queries, corpus, relevant_docs


# ===============================
# IR Metrics
# ===============================
def compute_recall_at_k(ranked, rel, k):
    return 1.0 if any(doc in rel for doc in ranked[:k]) else 0.0

def compute_mrr(ranked, rel):
    for i, d in enumerate(ranked):
        if d in rel:
            return 1.0 / (i + 1)
    return 0.0

def compute_dcg(ranked, rel, k):
    dcg = 0.0
    for i, d in enumerate(ranked[:k]):
        if d in rel:
            dcg += 1.0 / np.log2(i + 2)
    return dcg

def compute_ndcg_at_k(ranked, rel, k):
    dcg = compute_dcg(ranked, rel, k)
    ideal = compute_dcg(list(rel), rel, k)
    return 0.0 if ideal == 0 else dcg / ideal


# ===============================
# Custom IR Evaluator
# ===============================
class JasperIREvaluator(SentenceEvaluator):
    def __init__(self, queries, corpus, relevant_docs, name="ir-eval"):
        self.queries = queries
        self.corpus = corpus
        self.relevant_docs = relevant_docs
        self.name = name

    def __call__(self, model, output_path=None, epoch=-1, steps=-1):

        start_time = time.time()

        corpus_ids = list(self.corpus.keys())
        corpus_emb = model.encode(
            [self.corpus[cid] for cid in corpus_ids],
            prompt_name="query",
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        query_ids = list(self.queries.keys())
        query_emb = model.encode(
            [self.queries[qid] for qid in query_ids],
            prompt_name="query",
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        corpus_emb_T = corpus_emb.T

        mrrs, r10s, ndcgs = [], [], []

        for i, qid in enumerate(query_ids):
            sims = query_emb[i] @ corpus_emb_T
            ranked_idx = sims.argsort()[::-1]
            ranked_docs = [corpus_ids[j] for j in ranked_idx]

            rel = self.relevant_docs[qid]

            mrrs.append(compute_mrr(ranked_docs, rel))
            r10s.append(compute_recall_at_k(ranked_docs, rel, 10))
            ndcgs.append(compute_ndcg_at_k(ranked_docs, rel, 10))

        MRR = float(np.mean(mrrs))
        R10 = float(np.mean(r10s))
        N10 = float(np.mean(ndcgs))

        # JSON logging (separate file)
        eval_logger.write({
            "type": "eval",
            "epoch": float(epoch),
            "step": steps,
            "mrr": MRR,
            "recall_10": R10,
            "ndcg_10": N10,
            "duration": time.time() - start_time,
            "time": time.time()
        })

        print(f"[IR Eval epoch={epoch} step={steps}] "
              f"MRR={MRR:.4f} | R@10={R10:.4f} | nDCG@10={N10:.4f}")

        return MRR


# ===============================
# Load training samples
# ===============================
def load_pairs(path: Path):
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            q = (obj.get("query") or "").strip()
            p = (obj.get("passage") or "").strip()
            if q and p:
                samples.append(InputExample(texts=[q, p]))
    random.shuffle(samples)
    return samples


# ===============================
# Main Training Loop
# ===============================
def main():
    setup_logging()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True, device=device)
    model.max_seq_length = 512

    all_samples = load_pairs(DATA_PATH)
    print("Loaded samples:", len(all_samples))

    dev_size = int(len(all_samples) * 0.1)
    dev_samples = all_samples[:dev_size]
    train_samples = all_samples[dev_size:]

    train_loader = DataLoader(train_samples, batch_size=8, shuffle=True, drop_last=True)

    train_loss = losses.MultipleNegativesRankingLoss(model)

    queries, corpus, relevant_docs = build_ir_dev_set(dev_samples)
    evaluator = JasperIREvaluator(queries, corpus, relevant_docs)

    num_epochs = 1
    warmup_steps = int(len(train_loader) * num_epochs * 0.1)
    evaluation_steps = max(10, 1000 // 8)

    print("Warmup steps:", warmup_steps)
    print("Eval every", evaluation_steps, "steps")

    # -------------------------------
    # Loss logging (reduced frequency)
    # -------------------------------
    LOG_LOSS_EVERY = 20
    old_forward = train_loss.forward

    def loss_with_logging(*args, **kwargs):
        loss_value = old_forward(*args, **kwargs)
        if loss_logger.step % LOG_LOSS_EVERY == 0:
            loss_logger.write({
                "type": "train",
                "step": loss_logger.step,
                "loss": float(loss_value.item()),
                "time": time.time()
            })
        loss_logger.step += 1
        return loss_value

    train_loss.forward = loss_with_logging

    # -------------------------------
    # Train
    # -------------------------------
    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        use_amp=False,
        output_path=OUTPUT_DIR,
        evaluator=evaluator,
        evaluation_steps=evaluation_steps,
        save_best_model=True,
    )

    print("Finished! Best model saved to:", OUTPUT_DIR)
    print("Loss logs:", LOG_TRAIN_LOSS)
    print("Eval logs:", LOG_EVAL_METRICS)


if __name__ == "__main__":
    main()

