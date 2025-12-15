from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm

def load_qwen3_st(model_name="Qwen3-Embedding-4B", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Using device:", device)

    model = SentenceTransformer(
        model_name,
        model_kwargs={
            "device_map": "cuda",     
            "trust_remote_code": True,
            "attn_implementation": None,   
        },
        tokenizer_kwargs={
            "padding_side": "left",  
            "trust_remote_code": True,
        },
        trust_remote_code=True,
        local_files_only=True
        
    )
    model.max_seq_length = 512

    def encode(texts, batch_size=32, prompt_name=None):
        all_emb = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = texts[i:i+batch_size]

            emb = model.encode(
                batch,
                batch_size=batch_size,
                convert_to_tensor=True,
                show_progress_bar=False,
                normalize_embeddings=True,     # IMPORTANT for retrieval
                prompt_name=prompt_name,
            )

            all_emb.append(emb.cpu())

        return torch.cat(all_emb, dim=0)


    model.encode_cuda = encode
    return model



def load_gte_qwen2_st(
    model_name_or_dir="Alibaba-NLP/gte-Qwen2-1.5B-instruct", 
    device=None,
    max_seq_length=512,  
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Using device:", device)

    model = SentenceTransformer(
        model_name_or_dir,
        model_kwargs={
            "device_map": "cuda",          
            "trust_remote_code": True,
            "attn_implementation": None,
        },
        tokenizer_kwargs={
            "padding_side": "left",      
            "trust_remote_code": True,
        },
        trust_remote_code=True,
        local_files_only=True,          
    )

    model.max_seq_length = max_seq_length
    model._first_module().auto_model.config.use_cache = False

    def encode(texts, batch_size=128, prompt_name=None):
        all_emb = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = texts[i:i + batch_size]

            emb = model.encode(
                batch,
                batch_size=batch_size,
                convert_to_tensor=True,
                show_progress_bar=False,
                normalize_embeddings=True,  # 检索必备
                prompt_name=prompt_name,    # query 用 "query"，doc 用 None
            )
            all_emb.append(emb.cpu())

        return torch.cat(all_emb, dim=0)

    model.encode_cuda = encode
    return model



def load_jasper_st(
    model_name="Jasper-Token-Compression-600M",
    device=None,
    default_compression_ratio=0.3333,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Using device:", device)

    model = SentenceTransformer(
        model_name,
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "sdpa",
            "trust_remote_code": True,
            "device_map": "cuda",
        },
        tokenizer_kwargs={
            "padding_side": "left",
            "trust_remote_code": True,
        },
        trust_remote_code=True,
        local_files_only=True,
        device=device,
    )

    model.max_seq_length = 512

    def encode(texts, batch_size=32, prompt_name=None, compression_ratio=None):
        if compression_ratio is None:
            compression_ratio = default_compression_ratio

        all_emb = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = texts[i:i + batch_size]

            emb = model.encode(
                batch,
                batch_size=batch_size,
                convert_to_tensor=True,
                show_progress_bar=False,
                normalize_embeddings=True,
                prompt_name=prompt_name,
                compression_ratio=compression_ratio,
            )
            all_emb.append(emb.cpu())

        return torch.cat(all_emb, dim=0)

    model.encode_cuda = encode
    return model


def load_mongodb_st(
    model_name="MongoDB",
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Using device:", device)

    model = SentenceTransformer(
        model_name,
        model_kwargs={
            "device_map": "cuda", 
            "trust_remote_code": True,
           
        },
        tokenizer_kwargs={
            "padding_side": "left",
            "trust_remote_code": True,
        },
        trust_remote_code=True,
        local_files_only=True,           
    )

    model.max_seq_length = 512 
    def encode(texts, batch_size=128, prompt_name=None):
        all_emb = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = texts[i:i + batch_size]

            emb = model.encode(
                batch,
                batch_size=batch_size,
                convert_to_tensor=True,
                show_progress_bar=False,
                normalize_embeddings=True, 
                prompt_name=prompt_name, 
            )
            all_emb.append(emb.cpu())

        return torch.cat(all_emb, dim=0)

    model.encode_cuda = encode
    return model



from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
from tqdm import tqdm

def load_f2llm(
    model_dir="F2LLM", 
    device=None,
    max_length=512,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
    )

    model = AutoModel.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="cuda" if device == "cuda" else {"": 0},
        local_files_only=True,
    )
    model.eval()

    def encode(texts, batch_size=128, prompt_name=None):
        all_emb = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = texts[i:i + batch_size]
            batch = [s + tokenizer.eos_token for s in batch]

            tokenized_inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
                add_special_tokens=False,
            ).to(model.device)

            with torch.no_grad():
                outputs = model(**tokenized_inputs)
                last_hidden_state = outputs.last_hidden_state

            eos_positions = tokenized_inputs.attention_mask.sum(dim=1) - 1
            embeddings = last_hidden_state[
                torch.arange(len(batch), device=model.device), eos_positions
            ]

            # L2 normalize
            embeddings = F.normalize(embeddings, p=2, dim=1)

            all_emb.append(embeddings.cpu())

        return torch.cat(all_emb, dim=0)

    model.encode_cuda = encode
    model.f2llm_tokenizer = tokenizer  
    return model


def load_yuan_st(
    model_name="IEITYuan/Yuan-embedding-2.0-en", 
    device=None,
    max_seq_length=512,
):
   
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model = SentenceTransformer(
        model_name,
        model_kwargs={
            "device_map": "cuda" if device == "cuda" else None,
            "trust_remote_code": True,
        },
        tokenizer_kwargs={
            "padding_side": "left",
            "trust_remote_code": True,
        },
        trust_remote_code=True,
        local_files_only=True,  
        device=device,       
    )

    model.max_seq_length = max_seq_length

    def encode(texts, batch_size=128, prompt_name=None):
       
        all_emb = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = texts[i:i + batch_size]

            emb = model.encode(
                batch,
                batch_size=len(batch),     
                convert_to_tensor=True,
                show_progress_bar=False,
                normalize_embeddings=True,
            
                **({"prompt_name": prompt_name} if prompt_name is not None else {})
            )


            all_emb.append(emb.cpu())

        if len(all_emb) == 0:
            return torch.empty((0, model.get_sentence_embedding_dimension()))
        return torch.cat(all_emb, dim=0)

    model.encode_cuda = encode
    return model



def load_corpus(corpus_path):
    corpus = {}
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            doc_id = item["document_id"] if "document_id" in item else item["_id"]
            title = item.get("title", "")
            text = item.get("text", "")
            corpus[doc_id] = {
                "title": title,
                "text": text,
            }
    return corpus

def load_queries(query_path):
    queries = {}
    with open(query_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            qid = item.get("_id") or item.get("query_id") or item["task_id"]
            text = item["text"]
            queries[qid] = text
    return queries


def run_retrieval_for_collection(name, cfg, model, top_k=50):
    import os, torch
    from tqdm import tqdm

    root = cfg["root"]
    corpus_path = os.path.join(root, cfg["corpus_file"])
    query_path = os.path.join(root, cfg["query_file"])

    print(f"\n========== Collection: {name} ==========")
    print("Corpus:", corpus_path)
    print("Queries:", query_path)

    # Load data
    corpus = load_corpus(corpus_path)
    queries = load_queries(query_path)

    # ------------------------------
    # Encode documents (NO PROMPT)
    # ------------------------------
    doc_ids = list(corpus.keys())
    doc_texts = [
        (corpus[d].get("title", "") + " " + corpus[d].get("text", "")).strip()
        for d in doc_ids
    ]

    doc_emb = model.encode_cuda(doc_texts, batch_size=256, prompt_name=None)
    print("doc_emb shape:", doc_emb.shape)

    # ------------------------------
    # Encode queries (WITH Qwen3 query prompt)
    # ------------------------------
    q_ids = list(queries.keys()) 
    q_texts = [queries[q] for q in q_ids]

    q_emb = model.encode_cuda(q_texts, batch_size=256, prompt_name="query")
    print("q_emb shape:", q_emb.shape)

    # ------------------------------
    # Cosine similarity (already normalized)
    # ------------------------------
    sims = torch.matmul(q_emb, doc_emb.T)

    results = []
    for qi, qid in enumerate(tqdm(q_ids, desc="Building results")):
        vals, idx = torch.topk(sims[qi], top_k)
        ctxs = [
            {"document_id": doc_ids[j], "score": float(v)}
            for v, j in zip(vals.tolist(), idx.tolist())
        ]

        results.append({
            "task_id": qid,
            "contexts": ctxs,
            "Collection": cfg["collection_name"],
        })

    return results



def build_submission(submission_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = load_qwen3_st(model_name="Qwen3-Embedding-4B", device=device)
    # model = load_jasper_st(
    #     model_name="jasper-token-compression-600M-rag-ft-old-1-epoch",
    #     device=device,
    #     default_compression_ratio=0.3333,  
    # )
    # model = load_mongodb_st(
    #     model_name="MongoDB", 
    #     device=device,
    # )
    # model = load_f2llm(
    #     model_dir="F2LLM-1.7B",  
    #     device=device,
    #     max_length=512,
    # )
    model = load_gte_qwen2_st(
        model_name_or_dir="gte-Qwen2-1.5B-instruct", 
        device=device,
        max_seq_length=512,  
    )
    
    # model = load_yuan_st(
    #     model_name="Yuan-embedding-2.0-en", 
    #     device=device,
    #     max_seq_length=512,
    # )

    all_results = []
    for name, cfg in COLLECTIONS.items():
        print("Current name is", name)
        res = run_retrieval_for_collection(name, cfg, model, top_k=50)
        all_results.extend(res)

    print(f"\nTotal tasks: {len(all_results)}")
    with open(submission_path, "w", encoding="utf-8") as f:
        for item in all_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print("Saved submission to:", submission_path)
    return submission_path


import subprocess
import os

def run_official_eval(input_file, output_file):
    cmd = [
        "python3",
        "scripts/evaluation/run_retrieval_eval.py",
        "--input_file", input_file,
        "--output_file", output_file,
        "--model_name", model_name,
        "--task_name", task_name
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("Done, scored file:", output_file)



import os, json, torch
from tqdm import tqdm

for task_name in ['questions']:
    model_name = "gte_1.5B"

    COLLECTIONS = {
        "clapnq": {
            "collection_name": "mt-rag-clapnq-elser-512-100-20240503",
            "root": "human/retrieval_tasks/clapnq",
            "corpus_file": "clapnq.jsonl",
            "query_file": f"clapnq_{task_name}.jsonl",
            "qrels_file": "qrels/dev.tsv",
        },
        "fiqa": {
            "collection_name": "mt-rag-fiqa-beir-elser-512-100-20240501",
            "root": "human/retrieval_tasks/fiqa",
            "corpus_file": "fiqa.jsonl",
            "query_file": f"fiqa_{task_name}.jsonl",
            "qrels_file": "qrels/dev.tsv",
        },
        "govt": {
            "collection_name": "mt-rag-govt-elser-512-100-20240611",
            "root": "human/retrieval_tasks/govt",
            "corpus_file": "govt.jsonl",
            "query_file": f"govt_{task_name}.jsonl",
            "qrels_file": "qrels/dev.tsv",
        },
        "cloud": {
            "collection_name": "mt-rag-ibmcloud-elser-512-100-20240502",
            "root": "human/retrieval_tasks/cloud",
            "corpus_file": "cloud.jsonl",
            "query_file": f"cloud_{task_name}.jsonl",
            "qrels_file": "qrels/dev.tsv",
        },
    }


    SUB_PATH = f"outputs/{model_name}_{task_name}.jsonl"
    SCORED_PATH = f"outputs/{model_name}_{task_name}_score.jsonl"

    print("Start!")
    print("Current task", task_name)
    sub_path = build_submission(SUB_PATH)
    run_official_eval(SUB_PATH, SCORED_PATH)
