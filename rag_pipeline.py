import os, os.path as op, glob
import re
import json
from typing import List, Tuple, Dict, Any

from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma

from flashrank import Ranker, RerankRequest
from groq import Groq

import re
import nltk
from typing import List, Dict, Set, Tuple
from collections import Counter
import string

import warnings
warnings.filterwarnings('ignore')



EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_BATCH = 32
FLASHRANK_MODEL = "ms-marco-MiniLM-L-12-v2"
COLLECTION_NAME = "sec_filings"
NOT_FOUND_MSG = "This question cannot be answered based on the provided documents."
GENERATIVE_MODEL = "openai/gpt-oss-20b"
TOP_K = 10
PREFETCH = 30
CHUNK_SIZE = 700
CHUNK_OVERLAP = 150


SEC_ITEM_PATTERNS = [
    r"Item\s+(\d+(?:\.\d+)?(?:[A-Z])?)\s*[\.:]?\s*(.{1,100}?)(?:\n|\r|$)",
    r"ITEM\s+(\d+(?:\.\d+)?(?:[A-Z])?)\s*[\.:]?\s*(.{1,100}?)(?:\n|\r|$)",
    r"Part\s+([IV]+)\s*[,\-]\s*Item\s+(\d+(?:\.\d+)?(?:[A-Z])?)\s*[\.:]?\s*(.{1,100}?)(?:\n|\r|$)",
]




USER_PROMPT_TEMPLATE = """
You are an SEC filing analyst. Use ONLY the passages below.

IMPORTANT: Do NOT use table of contents, index pages, or navigation sections to answer questions. Only use actual content sections with substantive information.

TASK:
1) Answer the question in ≤ 25 words.
2) Only respond in below json structure. Don't respond with anything else.
{{"answer": "<actual_answer>", "ref_ids": "<list of valid chunk ids>", "reason": "<reason for answer>"}}
If the answer is missing or ambiguous, return:
{{"answer":"{not_found_msg}","ref_ids":[], "reason": "<reason for no answer>"}}


INSTRUCTIONS:
1) Generally there is only one valid chunk id.
2) Always consider signing date of the document as filing date of the document with the SEC event though not explicitly specified.
3) Don't give vauge answers, provide acutal value and what and for what like provide full context in the answer.
4) Review the dates given by users correctly then only answer.
5) When providing figures, always include denominations (like $) and units (like million or billion).
6) Only state facts provided in passages. If can't be answered respond accordigly.
7) Follow all the above mentioned instructions.

VALID CHUNK_IDs: {valid_ids}

QUESTION:
{query}

PASSAGES:
{context}

Return ONLY the JSON object on a single line. No extra text. No exceptions.
Only respond in JSON format.
""".strip()

SYSTEM_PROMPT = "Return strict, valid JSON only. No explanations. Only respond in JSON format. No exceptions. Always follow all the instructions mentioned by the user."



questions = [
{"question_id": 1, "question": "What was Apples total revenue for the fiscal year ended September 28, 2024?"},
{"question_id": 2, "question": "How many shares of common stock were issued and outstanding as of October 18, 2024?"},
{"question_id": 3, "question": "What is the total amount of term debt (current + non-current) reported by Apple as of September 28, 2024?"},
{"question_id": 4, "question": "On what date was Apples 10-K report for 2024 signed and filed with the SEC?"},
{"question_id": 5, "question": "Does Apple have any unresolved staff comments from the SEC as of this filing? How do you know?"},
{"question_id": 6, "question": "What was Teslas total revenue for the year ended December 31, 2023?"},
{"question_id": 7, "question": "What percentage of Teslas total revenue in 2023 came from Automotive Sales (excluding Leasing)?"},
{"question_id": 8, "question": "What is the primary reason Tesla states for being highly dependent on Elon Musk?"},
{"question_id": 9, "question": "What types of vehicles does Tesla currently produce and deliver?"},
{"question_id": 10, "question": "What is the purpose of Teslas ’lease pass-through fund arrangements’?"},
{"question_id": 11, "question": "What is Teslas stock price forecast for 2025?"},
{"question_id": 12, "question": "Who is the CFO of Apple as of 2025?"},
{"question_id": 13, "question": "What color is Teslas headquarters painted?"}
]





# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class StopwordKeywordReranker:
    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2"):
        self.ranker = Ranker(model_name=model_name)
        
        # Extended stopwords for financial documents
        self.english_stopwords = set(stopwords.words('english'))
        
        # Add SEC/Financial document common stopwords
        self.financial_stopwords = {
            'company', 'companies', 'business', 'operations', 'period', 
            'year', 'years', 'quarter', 'quarters', 'fiscal', 'ended',
            'including', 'related', 'certain', 'various', 'primarily',
            'approximately', 'substantially', 'significantly', 'generally',
            'may', 'could', 'would', 'should', 'might', 'will', 'shall',
            'also', 'however', 'therefore', 'furthermore', 'moreover',
            'item', 'part', 'section', 'table', 'note', 'see', 'refer'
        }
        
        self.all_stopwords = self.english_stopwords.union(self.financial_stopwords)
        
        # Important financial/SEC keywords that should NEVER be removed
        self.protected_keywords = {
            'revenue', 'earnings', 'profit', 'loss', 'cash', 'debt', 'assets',
            'liabilities', 'equity', 'shares', 'dividend', 'eps', 'ebitda',
            'operating', 'income', 'expenses', 'margin', 'growth', 'risk',
            'material', 'adverse', 'segment', 'goodwill', 'impairment',
            'depreciation', 'amortization', 'taxes', 'automotive', 'interest', 'cost',
            'sales', 'services', 'products', 'customers', 'market', 'competition',
            'regulatory', 'compliance', 'litigation', 'contingencies',
            'apple', 'tesla', 'automotive', 'technology', 'manufacturing', 'unresolved', 'leasing'
        }
    
    def _clean_and_tokenize(self, text: str) -> List[str]:
        """Clean text and tokenize while preserving important terms"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation but preserve $ and % symbols
        text = re.sub(r'[^\w\s\$%]', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords but preserve protected keywords and numbers
        cleaned_tokens = []
        for token in tokens:
            # Keep if it's a protected keyword
            if token in self.protected_keywords:
                cleaned_tokens.append(token)
            # Keep if it's a number (including financial numbers)
            elif re.match(r'^\d+(?:\.\d+)?$', token):
                cleaned_tokens.append(token)
            # Keep if it has $ or %
            elif '$' in token or '%' in token:
                cleaned_tokens.append(token)
            # Keep if it's not a stopword and is long enough
            elif token not in self.all_stopwords and len(token) > 2:
                cleaned_tokens.append(token)
        
        return cleaned_tokens
    
    def _extract_financial_patterns(self, text: str) -> List[str]:
        """Extract specific financial patterns"""
        patterns = []
        
        # Dollar amounts
        dollar_patterns = re.findall(r'\$\s*\d+(?:\.\d+)?(?:\s*(?:million|billion|trillion))?', text.lower())
        patterns.extend(dollar_patterns)
        
        # Percentages
        percent_patterns = re.findall(r'\d+(?:\.\d+)?%', text)
        patterns.extend(percent_patterns)
        
        # Years
        year_patterns = re.findall(r'\b(?:19|20)\d{2}\b', text)
        patterns.extend(year_patterns)
        
        # SEC items
        item_patterns = re.findall(r'\bitem\s+\d+[a-z]?\b', text.lower())
        patterns.extend(item_patterns)
        
        return patterns
    
    def _calculate_keyword_similarity(self, query_tokens: List[str], doc_tokens: List[str], 
                                    query_patterns: List[str], doc_patterns: List[str]) -> Dict[str, float]:
        """Calculate various keyword similarity metrics"""
        
        # 1. Token overlap score
        query_set = set(query_tokens)
        doc_set = set(doc_tokens)
        
        if not query_set:
            return {'token_overlap': 0.0, 'pattern_match': 0.0, 'jaccard': 0.0, 'weighted_score': 0.0}
        
        intersection = query_set.intersection(doc_set)
        token_overlap = len(intersection) / len(query_set)
        
        # 2. Financial pattern matching
        pattern_matches = 0
        for pattern in query_patterns:
            if pattern in doc_patterns:
                pattern_matches += 1
        
        pattern_match = pattern_matches / max(1, len(query_patterns))
        
        # 3. Jaccard similarity
        union = query_set.union(doc_set)
        jaccard = len(intersection) / len(union) if union else 0.0
        
        # 4. Weighted score (emphasize protected keywords)
        weighted_intersection = 0
        for token in intersection:
            if token in self.protected_keywords:
                weighted_intersection += 2.0  # Double weight for important terms
            else:
                weighted_intersection += 1.0
        
        weighted_score = weighted_intersection / len(query_set)
        
        return {
            'token_overlap': token_overlap,
            'pattern_match': pattern_match,
            'jaccard': jaccard,
            'weighted_score': weighted_score
        }
    
    def _filter_by_keywords(self, query: str, docs: List[Document], 
                           min_similarity: float = 0.1) -> List[Tuple[Document, Dict[str, float]]]:
        """Filter and score documents based on keyword similarity"""
        
        # Process query
        query_tokens = self._clean_and_tokenize(query)
        query_patterns = self._extract_financial_patterns(query)
        
        # print(f"[keyword] Query tokens after cleaning: {query_tokens}")
        # print(f"[keyword] Query patterns: {query_patterns}")
        
        scored_docs = []
        
        for doc in docs:
            # Process document
            doc_tokens = self._clean_and_tokenize(doc.page_content)
            doc_patterns = self._extract_financial_patterns(doc.page_content)
            
            # Calculate similarity metrics
            similarity_metrics = self._calculate_keyword_similarity(
                query_tokens, doc_tokens, query_patterns, doc_patterns
            )
            
            # Combined keyword score
            combined_score = (
                0.4 * similarity_metrics['weighted_score'] +
                0.3 * similarity_metrics['token_overlap'] +
                0.2 * similarity_metrics['pattern_match'] +
                0.1 * similarity_metrics['jaccard']
            )
            
            # Only keep documents above minimum similarity
            if combined_score >= min_similarity:
                scored_docs.append((doc, {
                    **similarity_metrics,
                    'combined_keyword_score': combined_score
                }))
        
        # Sort by combined keyword score
        scored_docs.sort(key=lambda x: x[1]['combined_keyword_score'], reverse=True)
        
        # print(f"[keyword] Filtered from {len(docs)} to {len(scored_docs)} documents")
        
        return scored_docs
    
    def rerank(self, query: str, docs: List[Document], top_k: int = 5, 
               keyword_filter_threshold: float = 0.1,
               hybrid_weight: float = 0.3) -> List[Document]:
        """
        Three-stage reranking:
        1. Remove stopwords and filter by keyword similarity
        2. Apply FlashRank to filtered documents
        3. Combine keyword + FlashRank scores
        """
        if not docs:
            return []
        
        # print(f"[pipeline] Starting 3-stage reranking with {len(docs)} documents")
        
        # Stage 1: Keyword filtering
        keyword_scored_docs = self._filter_by_keywords(query, docs, keyword_filter_threshold)
        
        if not keyword_scored_docs:
            # print("[pipeline] No documents passed keyword filter, using all documents")
            keyword_scored_docs = [(doc, {'combined_keyword_score': 0.0}) for doc in docs]
        
        # Take top candidates for FlashRank (to avoid processing too many)
        candidates = keyword_scored_docs
        candidate_docs = [doc for doc, _ in candidates]
        
        # print(f"[pipeline] Stage 1 complete: {len(candidates)} candidates for FlashRank")
        
        # Stage 2: FlashRank reranking
        try:
            passages = [{"text": d.page_content} for d in candidate_docs]
            flashrank_results = self.ranker.rerank(RerankRequest(query=query, passages=passages))
            
            # Stage 3: Combine scores
            final_results = []
            
            for i, flashrank_item in enumerate(flashrank_results):
                # Use enumeration index since FlashRank returns dicts without index
                idx = i
                
                if idx >= len(candidates):
                    continue
                
                doc, keyword_metrics = candidates[idx]
                
                # Get FlashRank score from dict
                flashrank_score = flashrank_item.get('score', 0.0)
                
                # Normalize FlashRank score (typically -1 to 1 -> 0 to 1)
                normalized_flashrank = (flashrank_score + 1) / 2
                
                # Combine scores
                keyword_score = keyword_metrics['combined_keyword_score']
                final_score = (1 - hybrid_weight) * normalized_flashrank + hybrid_weight * keyword_score
                
                final_results.append({
                    'doc': doc,
                    'final_score': final_score,
                    'flashrank_score': flashrank_score,
                    'keyword_score': keyword_score,
                    'keyword_metrics': keyword_metrics
                })
            
            # Sort by final score
            final_results.sort(key=lambda x: x['final_score'], reverse=True)
            
            # Build output
            reranked_docs = []
            for i, result in enumerate(final_results[:top_k]):
                doc = result['doc']
                enhanced_metadata = doc.metadata.copy()
                enhanced_metadata['rerank_score'] = result['final_score']
                enhanced_metadata['flashrank_score'] = result['flashrank_score']
                enhanced_metadata['keyword_score'] = result['keyword_score']
                enhanced_metadata.update(result['keyword_metrics'])
                
                # print(f"[pipeline] Doc {i+1}: Final={result['final_score']:.3f} "
                #       f"(FlashRank={result['flashrank_score']:.3f}, "
                #       f"Keyword={result['keyword_score']:.3f})")
                
                reranked_docs.append(Document(page_content=doc.page_content, metadata=enhanced_metadata))
            
            return reranked_docs
            
        except Exception as e:
            print(f"[pipeline] FlashRank error: {e}")
            print("[pipeline] Falling back to keyword-only ranking")
            
            # Fallback: Use only keyword scores
            fallback_docs = []
            for i, (doc, metrics) in enumerate(candidates[:top_k]):
                enhanced_metadata = doc.metadata.copy()
                enhanced_metadata['rerank_score'] = metrics['combined_keyword_score']
                enhanced_metadata['keyword_score'] = metrics['combined_keyword_score']
                enhanced_metadata.update(metrics)
                
                fallback_docs.append(Document(page_content=doc.page_content, metadata=enhanced_metadata))
            
            return fallback_docs

# Usage in your existing pipeline
def retrieve_sec_info_enhanced(vectorstore: Chroma, query: str, k: int = 5, prefetch: int = PREFETCH) -> List[Dict]:
    """Enhanced retrieval with stopword removal and keyword matching"""
    print(f"[retrieve] Enhanced query: {query}")

    # Step 1: Initial vector search (larger prefetch for filtering)
    candidates = vectorstore.similarity_search(query, k=prefetch)  # Get more candidates
    print(f"[retrieve] Retrieved {len(candidates)} initial candidates")

    # Step 2: Enhanced reranking with stopword removal and keyword matching
    enhanced_reranker = StopwordKeywordReranker("ms-marco-MiniLM-L-12-v2")
    
    top_docs = enhanced_reranker.rerank(
        query, 
        candidates, 
        top_k=k,
        keyword_filter_threshold=0.05,  # Lower threshold = more permissive
        hybrid_weight=0.4  # 40% keyword weight, 60% FlashRank weight
    )
    
    print(f"[retrieve] Final reranked results: {len(top_docs)}")

    # Convert to your expected format
    results = []
    for doc in top_docs:
        metadata = doc.metadata or {}
        results.append({
            'chunk_id': metadata.get('chunk_id', ''),
            'content': doc.page_content,
            'document_name': metadata.get('document_name', ''),
            'page_number': metadata.get('page', 0),
            'item_number': metadata.get('item_number', ''),
            'item_title': metadata.get('item_title', ''),
            'rerank_score': metadata.get('rerank_score', 0.0),
            # Additional keyword metrics
            'keyword_score': metadata.get('keyword_score', 0.0),
            'flashrank_score': metadata.get('flashrank_score', 0.0),
            'token_overlap': metadata.get('token_overlap', 0.0),
            'pattern_match': metadata.get('pattern_match', 0.0),
        })
    
    return results

class embeddingclass(Embeddings):
    def __init__(self, model_name: str = EMBED_MODEL, device: str = DEVICE, load_model: bool = True):
        self.model_name = model_name
        self.device = device
        self.tok = None
        self.model = None
        if load_model:
            self.load_model()

    def load_model(self):
        if self.model is None:
            # print(f"Loading model on {self.device}")
            self.tok = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            if self.tok.pad_token_id is None:
                self.tok.pad_token = self.tok.eos_token
            
            # Load model with better GPU settings
            self.model = AutoModel.from_pretrained(
                self.model_name, 
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map=None
            ).to(self.device)
            
            # Set to eval mode and optimize for inference
            self.model.eval()
            if self.device == "cuda":
                torch.set_float32_matmul_precision("high")
                # # Compile model for better GPU performance (PyTorch 2.0+)
                # try:
                #     self.model = torch.compile(self.model, mode="reduce-overhead")
                # except:
                #     pass  # fallback if compile not available

    @torch.no_grad()
    def _encode_batch(self, texts: List[str]) -> List[List[float]]:
        if self.model is None:
            self.load_model()
        
        # Tokenize and move to GPU immediately
        t_cpu = self.tok(
            texts, 
            padding="longest", 
            truncation=True, 
            max_length=1024, 
            return_tensors="pt"
        )
        
        # Move to GPU with non_blocking for better performance
        if self.device == "cuda":
            t = {k: v.to(self.device, non_blocking=True) for k, v in t_cpu.items()}
        else:
            t = t_cpu
            
        # Use autocast for better GPU performance
        autocast_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16) if self.device == "cuda" else contextlib.nullcontext()
        
        with autocast_ctx:
            # Forward pass - should use GPU
            out = self.model(**t).last_hidden_state
            
            # Pooling operations on GPU
            mask = t["attention_mask"].unsqueeze(-1).expand_as(out).float()
            emb = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        
        # Only convert to CPU at the very end
        return emb.float().cpu().tolist()

    def embed_documents(self, texts: List[str], chunk_size: int = 0) -> List[List[float]]:
        embs: List[List[float]] = []
        bs = EMBED_BATCH if chunk_size == 0 else chunk_size
        
        print(f"Embedding {len(texts)} documents in batches of {bs} on {self.device}")
        
        for i in range(0, len(texts), bs):
            batch = texts[i:i+bs]
            batch_embs = self._encode_batch(batch)
            embs.extend(batch_embs)
            
            # Print progress and GPU usage
            if i % (bs * 4) == 0:  # every 4 batches
                if self.device == "cuda":
                    gpu_mem = torch.cuda.memory_allocated() / 1024**3
                    print(f"Processed {i+len(batch)}/{len(texts)} - GPU Memory: {gpu_mem:.2f}GB")
        
        return embs

    def embed_query(self, text: str) -> List[float]:
        return self._encode_batch([text])[0]

    async def aembed_documents(self, texts: List[str], chunk_size: int = 0) -> List[List[float]]:
        return self.embed_documents(texts, chunk_size)

    async def aembed_query(self, text: str) -> List[float]:
        return self.embed_query(text)

class rerankerclass:
    def __init__(self, model_name: str = FLASHRANK_MODEL):
        self.ranker = Ranker(model_name=model_name)

    def rerank(self, query: str, docs: List[Document], top_k: int = 5) -> List[Document]:
        if not docs:
            return []
        try:
            passages = [{"text": d.page_content} for d in docs]
            results = self.ranker.rerank(RerankRequest(query=query, passages=passages))
            reranked = []
            for item in results[:top_k]:
                idx = getattr(item, "index", getattr(item, "corpus_id", None))
                if idx is None and isinstance(item, dict):
                    idx = item.get("index", item.get("corpus_id", 0))
                if idx is not None and idx < len(docs):
                    doc = docs[idx]
                    md = dict(doc.metadata or {})
                    score = float(getattr(item, "score", item.get("score", 0.0))) if isinstance(item, dict) else float(getattr(item, "score", 0.0))
                    md["rerank_score"] = score
                    reranked.append(Document(page_content=doc.page_content, metadata=md))
            return reranked
        except Exception:
            out = []
            for doc in docs[:top_k]:
                md = dict(doc.metadata or {}); md["rerank_score"] = 0.0
                out.append(Document(page_content=doc.page_content, metadata=md))
            return out

class generationclass:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable required")
        self.client = Groq(api_key=self.api_key)

    @staticmethod
    def _pretty_filename_label(filename: str) -> str:
        return chromaclass._pretty_filename_label(filename)

    @staticmethod
    def _coerce_and_parse_json(raw: str) -> Dict[str, Any]:
        if not raw: return {}
        s = raw.strip()
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s); s = re.sub(r"\s*```$", "", s)
        s = s.replace(""","\"").replace(""","\"").replace("'","'").replace("'","'")
        b0, b1 = s.find("{"), s.rfind("}")
        if b0 != -1 and b1 != -1 and b1 > b0: s = s[b0:b1+1]
        s = re.sub(r",\s*([}\]])", r"\1", s)
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else {}
        except Exception: pass
        m_ans = re.search(r'"answer"\s*:\s*"([^"]*)"', s)
        m_ids = re.search(r'"ref_ids"\s*:\s*\[(.*?)\]', s, flags=re.DOTALL)
        ans = (m_ans.group(1).strip() if m_ans else "")
        ref_ids = re.findall(r'"([^"]+)"', m_ids.group(1)) if m_ids else []
        return {"answer": ans, "ref_ids": ref_ids} if (ans or ref_ids) else {}

    def _format_context(self, results: List[Dict]) -> str:
        blocks = []
        for r in results:
            cid = r.get("chunk_id",""); doc = r.get("document_name",""); page = r.get("page_number",""); item = r.get("item_number","")
            header = f"[CHUNK_ID:{cid}] [DOC:{doc}] [PAGE:{page}] [ITEM:{item}]"
            txt = r["content"][:CHUNK_SIZE] + ("..." if len(r["content"]) > CHUNK_SIZE else "")
            blocks.append(f"{header}\n{txt}")
        return "\n\n".join(blocks)

    def _chunk_ids_to_sources_flat(self, ref_ids: List[str], results: List[Dict]) -> List[str]:
        index = {r['chunk_id']: r for r in results if r.get('chunk_id')}
        out: List[str] = []
        for rid in ref_ids:
            r = index.get(rid)
            if not r: continue
            label = self._pretty_filename_label(r.get("document_name",""))
            page = int(r.get("page_number",0)) + 1
            item = r.get("item_number") or ""
            out.extend([label, f"Item {item}" if item else "Item", f"p. {page}"])
            break
        return out

    def generate(self, query: str, results: List[Dict]) -> Dict[str, Any]:
        if not results:
            return {"answer": NOT_FOUND_MSG, "sources": []}
        
        context = self._format_context(results)
        valid_ids = [r.get("chunk_id","") for r in results if r.get("chunk_id")]
        
        user_prompt = USER_PROMPT_TEMPLATE.format(
            not_found_msg=NOT_FOUND_MSG,
            valid_ids=", ".join(valid_ids),
            query=query,
            context=context
        )
        # print(user_prompt)

        completion = self.client.chat.completions.create(
            model=GENERATIVE_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=576,
        )
        
        raw = (completion.choices[0].message.content or "").strip()
        # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        # print(completion, 'completion')
        # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        obj = self._coerce_and_parse_json(raw)
        answer = (obj.get("answer") or "").strip()
        ref_ids = [rid for rid in (obj.get("ref_ids") or []) if rid in set([r.get("chunk_id","") for r in results])]
        
        if not answer or answer == NOT_FOUND_MSG:
            return {"answer": NOT_FOUND_MSG, "sources": []}
            
        return {"answer": answer, "sources": self._chunk_ids_to_sources_flat(ref_ids, results)}
class chromaclass:
    _vectorstore = None  # class-level singleton

    def __init__(self, pdf_dir: str = PDF_DIR, collection_name: str = COLLECTION_NAME, persist_dir: str = CHROMA_DB_PATH):
        self.pdf_dir = pdf_dir or os.environ.get("PDF_DIR", "/content")
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        if chromaclass._vectorstore is None:
            self._ensure_vectorstore()

    # ---- internal helpers kept inside the class ----
    @staticmethod
    def _sec_item_patterns() -> List[str]:
        return [
            r"Item\s+(\d+(?:\.\d+)?(?:[A-Z])?)\s*[\.:]?\s*(.{1,100}?)(?:\n|\r|$)",
            r"ITEM\s+(\d+(?:\.\d+)?(?:[A-Z])?)\s*[\.:]?\s*(.{1,100}?)(?:\n|\r|$)",
            r"Part\s+([IV]+)\s*[,\-]\s*Item\s+(\d+(?:\.\d+)?(?:[A-Z])?)\s*[\.:]?\s*(.{1,100}?)(?:\n|\r|$)",
        ]

    @classmethod
    def _extract_sec_items(cls, text: str, page_num: int) -> List[Dict[str, Any]]:
        items = []
        for pattern in cls._sec_item_patterns():
            for m in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
                groups = [g for g in m.groups() if g is not None]
                if not groups:
                    continue
                item_num, item_title = None, ""
                if len(groups) == 1:
                    pot = groups[0].strip()
                    if re.match(r'^[\dA-Z]+(\.\d+)?[A-Z]?$', pot):
                        item_num = pot
                elif len(groups) == 2:
                    a, b = groups
                    if re.match(r'^[\dA-Z]+(\.\d+)?[A-Z]?$', a.strip()):
                        item_num = a.strip(); item_title = b.strip()[:100]
                    elif re.match(r'^[\dA-Z]+(\.\d+)?[A-Z]?$', b.strip()):
                        if re.match(r'^(Part\s+)?[IVX]+$', a.strip(), re.IGNORECASE):
                            item_num = f"{a.strip()}-{b.strip()}"
                        else:
                            item_num = b.strip()
                elif len(groups) == 3:
                    part, pot, title = groups
                    if re.match(r'^[\dA-Z]+(\.\d+)?[A-Z]?$', pot.strip()):
                        item_num = f"{part.strip()}-{pot.strip()}"
                        item_title = (title or "").strip()[:100]
                if item_num:
                    items.append({'item_number': item_num, 'item_title': item_title, 'page': page_num, 'position': m.start()})
        seen, unique = set(), []
        for it in sorted(items, key=lambda x: x['position']):
            key = (it['item_number'], it['page'])
            if key not in seen:
                seen.add(key); unique.append(it)
        return unique

    @staticmethod
    def _assign_chunk_ids(split_docs: List[Document]) -> None:
        counters: Dict[Tuple[str, int], int] = {}
        for d in split_docs:
            md = d.metadata or {}
            fname = md.get("document_name") or op.basename(md.get("source", ""))
            page = int(md.get("page", 0))
            key = (fname, page)
            seq = counters.get(key, 0)
            d.metadata["chunk_id"] = f"{fname}|p{page}|c{seq}"
            counters[key] = seq + 1

    @staticmethod
    def _pretty_filename_label(filename: str) -> str:
        if not filename:
            return "Document"
        stem = op.splitext(op.basename(filename))[0]
        stem = re.sub(r"[_\-]+", " ", stem).strip()
        parts = []
        for w in stem.split():
            lw = w.lower()
            if lw in {"10k", "10-k"}: parts.append("10-K")
            elif lw in {"10q", "10-q"}: parts.append("10-Q")
            elif re.fullmatch(r"[A-Z0-9]{2,}", w): parts.append(w)
            else: parts.append(w.capitalize())
        return " ".join(parts) or "Document"

    def _collection_exists(self) -> bool:
        try:
            temp_emb = embeddingclass(load_model=False)
            vs = Chroma(
                collection_name=self.collection_name,
                embedding_function=temp_emb,
                persist_directory=self.persist_dir
            )
            hits = vs.similarity_search("test", k=1)
            return len(hits) > 0   # ← only “exists” if it actually has vectors
        except Exception:
            return False

    def _load_pdfs(self, paths: List[str]) -> List[Document]:
        all_docs: List[Document] = []
        for p in paths:
            loader = PyPDFLoader(p)
            page_docs = loader.load()
            fname = op.basename(p)
            page_docs.sort(key=lambda d: (d.metadata or {}).get("page", 0))
            current_items, last_item_page = [], 0
            for doc in page_docs:
                page_num = doc.metadata.get("page", 0)
                text = doc.page_content
                page_items = chromaclass._extract_sec_items(text, page_num)
                if page_items:
                    current_items.extend(page_items); current_items = current_items[-5:]; last_item_page = page_num
                primary_item = page_items[0] if page_items else (current_items[-1] if current_items and (page_num - last_item_page <= 5) else None)
                md = {
                    'source': doc.metadata.get('source', p),
                    'page': page_num,
                    'document_name': fname,
                    'has_sec_items': len(page_items) > 0,
                    'item_number': primary_item['item_number'] if primary_item else '',
                    'item_title': primary_item['item_title'] if primary_item else '',
                }
                all_docs.append(Document(page_content=doc.page_content, metadata=md))
        return all_docs

    def _build_index(self):
        pdfs = [p for p in glob.glob(op.join(self.pdf_dir, "*.pdf")) if op.isfile(p)]
        if not pdfs:
            raise FileNotFoundError(f"No PDF files found in {self.pdf_dir}")
        emb = embeddingclass(load_model=True)
        docs = self._load_pdfs(pdfs)
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separators=["\n\n", "\n", " ", ""])
        split_docs = splitter.split_documents(docs)
        chromaclass._assign_chunk_ids(split_docs)
        vs = Chroma.from_documents(documents=split_docs, embedding=emb, collection_name=self.collection_name, persist_directory=self.persist_dir)
        vs.persist()
        chromaclass._vectorstore = vs

    def _ensure_vectorstore(self):
        if self._collection_exists():
            chromaclass._vectorstore = Chroma(collection_name=self.collection_name, embedding_function=embeddingclass(load_model=False), persist_directory=self.persist_dir)
        else:
            self._build_index()

    def vectorstore(self) -> Chroma:
        return chromaclass._vectorstore

    def retrieve(self, query: str, k: int = 10, prefetch: int = 30) -> List[Dict[str, Any]]:
        candidates = self.vectorstore().similarity_search(query, k=prefetch)
        rr = rerankerclass()
        top_docs = rr.rerank(query, candidates, top_k=k) if candidates else []
        results = []
        for doc in top_docs:
            md = doc.metadata or {}
            results.append({
                'chunk_id': md.get('chunk_id',''),
                'content': doc.page_content,
                'document_name': md.get('document_name',''),
                'page_number': int(md.get('page',0)),
                'item_number': md.get('item_number',''),
                'item_title': md.get('item_title',''),
                'rerank_score': float(md.get('rerank_score',0.0)),
            })
        return results

    def retrieve_enhanced(self, query: str, k: int = 10, prefetch: int = 25) -> List[Dict[str, Any]]:
        """Enhanced retrieval with stopword removal and keyword matching"""
        
        # Get more initial candidates for better keyword filtering
        candidates = self.vectorstore().similarity_search(query, k=prefetch*2)

        # print(candidates)

        # for candidate in candidates:
        #     md = candidate.metadata 
        #     print(f'''"{md.get('chunk_id')}"''')
        
        # Use enhanced reranker
        enhanced_reranker = StopwordKeywordReranker("ms-marco-MiniLM-L-12-v2")
        
        top_docs = enhanced_reranker.rerank(
            query, 
            candidates, 
            top_k=k,
            keyword_filter_threshold=0.05,  # Adjust based on your needs
            hybrid_weight=0.3  # 30% keywords, 70% FlashRank
        )
        # print("**************")
        # print(top_docs)
        
        # Convert to your expected format
        results = []
        for doc in top_docs:
            md = doc.metadata or {}
            results.append({
                'chunk_id': md.get('chunk_id',''),
                'content': doc.page_content,
                'document_name': md.get('document_name',''),
                'page_number': int(md.get('page',0)),
                'item_number': md.get('item_number',''),
                'item_title': md.get('item_title',''),
                'rerank_score': float(md.get('rerank_score',0.0)),
                # Additional debugging info
                'keyword_score': float(md.get('keyword_score', 0.0)),
                'flashrank_score': float(md.get('flashrank_score', 0.0)),
            })
        return results


def answer_question(query: str) -> dict:
    """
    Answers a question using the RAG pipeline.
    Args:
        query (str): The user question about Apple or Tesla 10-K filings.
    Returns:
        dict: {
            "answer": "Answer text or ’This question cannot be answered based on the provided documents.’",
            "sources": ["Apple 10-K", "Item 8", "p. 28"]  # Empty list if refused
        }
    """
    store = chromaclass()               # builds/loads Chroma once (singleton)
    results = store.retrieve_enhanced(query, k=TOP_K, prefetch= 30 * 2)
    gen = generationclass()
    out = gen.generate(query, results)
    return {"answer": out.get("answer", NOT_FOUND_MSG), "sources": out.get("sources", [])}


for question in questions: 
    results = answer_question(question['question'])
    print(results)
    print("*"*50)
