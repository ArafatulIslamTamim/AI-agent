import os
import re
import tempfile
import hashlib
from typing import List, Dict, Tuple
from collections import defaultdict

import streamlit as st
from dotenv import load_dotenv
from langdetect import detect

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# ---------------- App & Env ----------------
st.set_page_config(page_title="RAG Chat", page_icon="🤖", layout="wide")
load_dotenv()

# ---------------- Defaults -----------------
CHUNK_SIZE = 1600
CHUNK_OVERLAP = 200
TOP_K = 8
MAX_SOURCE_CHARS = 9000  # LLM input budget

# Summary triggers (EN + BN)
SUMMARY_TRIGGERS = (
    "summarize", "summary", "overview", "tl;dr", "tldr",
    "what is the pdf about", "what the pdf about", "give me a summary",
    "সারসংক্ষেপ", "সার", "সংক্ষেপে", "ওভারভিউ", "সারাংশ", "পিডিএফটা কি সম্পর্কে", "পিডিএফ কি সম্পর্কে"
)

# Force RAG triggers (ask explicitly to answer from docs)
FORCE_RAG_TRIGGERS = (
    "from the pdf", "in the pdf", "from pdf", "cite", "source", "sources",
    "পিডিএফ থেকে", "ডকুমেন্ট থেকে", "উৎস", "রেফারেন্স", "পাতা", "পৃষ্ঠা", "page"
)

# Stopwords (minimal) for router
STOPWORDS_EN = {
    "the","a","an","and","or","of","to","in","for","on","at","by","with","from","as",
    "is","are","was","were","be","being","been","it","this","that","these","those",
    "about","over","into","than","then","so","such","not","can","could","should","would",
    "how","what","why","when","where","who","whom","which"
}
STOPWORDS_BN = {
    "এবং","কিন্তু","বা","একটি","টি","এই","ওই","এটা","সেটা","যা","যে","যাকে","যাদের",
    "কি","কী","কে","কেন","কখন","কোথায়","কিভাবে","কেমন","থেকে","জন্য","উপর","ভিতরে","বাইরে",
    "হয়","আছে","ছিল","থাকবে","হবে","না","আর","তাই","তো"
}

# ---------------- Key Gate -----------------
if "groq_key" not in st.session_state:
    st.session_state.groq_key = os.getenv("GROQ_API_KEY", "")

if not st.session_state.groq_key:
    st.title("🔑 Enter your Groq API Key")
    st.caption("No key found in environment. Create one: https://console.groq.com/keys")
    key_input = st.text_input("GROQ_API_KEY", type="password")
    if not key_input:
        st.stop()
    st.session_state.groq_key = key_input
    st.rerun()

groq_key = st.session_state.groq_key

# ---------------- Sidebar ------------------
with st.sidebar:
    st.title("⚙️ Settings")
    use_mmr = st.checkbox("Use MMR (diverse retrieval)", value=True)
    uploaded_files = st.file_uploader("Upload PDFs (auto-indexes)", type="pdf", accept_multiple_files=True)
    clear_btn = st.button("🗑️ Clear Index")

# ---------------- Session State ------------
if "vs" not in st.session_state:
    st.session_state.vs = None
if "all_chunks" not in st.session_state:
    st.session_state.all_chunks: List[Document] = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "upload_fingerprint" not in st.session_state:
    st.session_state.upload_fingerprint = ""

# ---------------- Caches -------------------
@st.cache_resource
def get_embeddings():
    # Multilingual embeddings (EN + BN)
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        encode_kwargs={"normalize_embeddings": True}
    )

@st.cache_resource
def make_llm(model_name: str, key: str):
    return ChatGroq(model=model_name, temperature=0.2, groq_api_key=key)

def invoke_with_fallback(llm: ChatGroq, messages, key: str):
    """Fallback between 8B and 70B if a model is unavailable."""
    try:
        return llm.invoke(messages)
    except Exception as e:
        msg = str(e).lower()
        if any(t in msg for t in ["decommissioned", "invalid model", "model_not_found", "unsupported"]):
            current = getattr(llm, "model", "") or getattr(llm, "model_name", "")
            alt = "llama-3.3-70b-versatile" if "8b" in current else "llama-3.1-8b-instant"
            st.warning(f"Model '{current}' unavailable; falling back to '{alt}'.")
            return ChatGroq(model=alt, temperature=0.2, groq_api_key=key).invoke(messages)
        raise

# ---------------- Ingestion ----------------
def fingerprint_uploads(files) -> str:
    """Stable fingerprint to detect when the uploaded set changes."""
    if not files:
        return ""
    h = hashlib.sha1()
    for f in files:
        try:
            size = len(f.getbuffer())
        except Exception:
            pos = f.tell(); data = f.read(); size = len(data); f.seek(pos)
        h.update(f.name.encode("utf-8")); h.update(str(size).encode("utf-8"))
    return h.hexdigest()

def process_pdfs(uploaded) -> Tuple[FAISS, List[Document]]:
    docs: List[Document] = []
    for f in uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.read()); tmp.flush()
            loader = PyPDFLoader(tmp.name)     # if BN text looks garbled, switch to PyMuPDFLoader
            pages = loader.load()              # one Document per page
            for d in pages:
                d.metadata["source"] = f.name
            docs.extend(pages)

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    vs = FAISS.from_documents(chunks, get_embeddings())
    return vs, chunks

def auto_index_if_needed():
    fp = fingerprint_uploads(uploaded_files)
    if clear_btn:
        st.session_state.vs = None
        st.session_state.all_chunks = []
        st.session_state.upload_fingerprint = ""
        st.sidebar.info("Index cleared.")
        return
    if uploaded_files and (fp != st.session_state.upload_fingerprint or st.session_state.vs is None):
        with st.spinner("Indexing PDFs..."):
            st.session_state.vs, st.session_state.all_chunks = process_pdfs(uploaded_files)
            st.session_state.upload_fingerprint = fp
            st.sidebar.success(f"Indexed {len(st.session_state.all_chunks)} chunks from {len(uploaded_files)} file(s).")

auto_index_if_needed()

# ---------------- Helpers ------------------
def detect_summary_intent(text: str) -> bool:
    tl = (text or "").lower()
    return any(t in tl for t in SUMMARY_TRIGGERS)

def wants_docs_explicitly(text: str) -> bool:
    tl = (text or "").lower()
    return any(t in tl for t in FORCE_RAG_TRIGGERS)

def detect_lang(text: str) -> str:
    try:
        return "bn" if detect((text or "").strip()) == "bn" else "en"
    except Exception:
        return "en"

def lang_name(code: str) -> str:
    return "Bengali" if code == "bn" else "English"

def sample_evenly(chunks: List[Document], per_file: int = 10) -> List[Document]:
    groups: Dict[str, List[Document]] = defaultdict(list)
    for d in chunks:
        groups[d.metadata.get("source", "unknown")].append(d)
    sampled: List[Document] = []
    for _, docs in groups.items():
        docs.sort(key=lambda d: d.metadata.get("page", 0))
        n = min(per_file, len(docs))
        if n <= 0: 
            continue
        step = len(docs) / n
        idxs = sorted({min(int(i * step), len(docs)-1) for i in range(n)})
        sampled.extend(docs[i] for i in idxs)
    return sampled

def format_sources_for_prompt(docs: List[Document], max_chars: int = MAX_SOURCE_CHARS) -> str:
    parts, total = [], 0
    for d in docs:
        s = f"[{d.metadata.get('source','?')} | p.{d.metadata.get('page','?')}] {d.page_content.strip()}"
        if total + len(s) > max_chars:
            break
        parts.append(s); total += len(s)
    return "\n\n".join(parts)

def keyword_overlap_router(query: str, docs: List[Document]) -> bool:
    """
    If none of the query's keywords appear in retrieved text, treat as unrelated → fall back to chat.
    """
    q = (query or "").lower()
    q_terms_en = set(re.findall(r"\b[a-z0-9]{3,}\b", q)) - STOPWORDS_EN
    q_terms_bn = set(re.findall(r"[\u0980-\u09FF]{2,}", q)) - STOPWORDS_BN
    q_terms = q_terms_en | q_terms_bn
    if not q_terms:
        return False
    joined = " ".join(d.page_content.lower() for d in docs)[:80000]
    hits = sum(1 for t in q_terms if t in joined)
    return hits == 0

def looks_like_nonanswer(text: str) -> bool:
    t = (text or "").strip().lower()
    if len(t) < 20: return True
    triggers = [
        "don’t have enough information","don't have enough information",
        "i don't have enough information","insufficient information",
        "not enough information","cannot determine from the brief",
        "তথ্য পর্যাপ্ত নয়","পর্যাপ্ত তথ্য নেই","উপলব্ধ তথ্য যথেষ্ট নয়"
    ]
    return any(x in t for x in triggers)

# ---------------- Prompts -------------------
SYSTEM_SUMMARY = """You are a helpful assistant that writes a concise, faithful brief from provided sources.
- Summarize only what appears in the sources.
- Prefer concepts/explanations over raw symbol lists.
- If you mention time complexity, explain what each Big-O measures.
- Use bullet points and short paragraphs.
- Do not invent facts.
Respond in {lang_name}."""
HUMAN_SUMMARY = """User Query:
{query}

Sources (each begins with [file | page]):
{sources}

Write a concise, query-focused brief (<= 200 words) in {lang_name}."""

SYSTEM_QA = """You answer strictly based on the provided brief.
- If the brief lacks the answer, say you don’t have enough information.
- Be precise and succinct.
Respond in {lang_name}."""
HUMAN_QA = """User Query:
{query}

Brief (ground truth from the documents):
{brief}

Answer only from the brief in {lang_name}. Keep it under ~150 words."""

# ---------------- Small helpers (LLM calls) ---------------
def llm_reply(prompt_text: str) -> str:
    llm = make_llm("llama-3.1-8b-instant", groq_key)
    sys = "You are a helpful, concise assistant."
    tmpl = ChatPromptTemplate.from_messages([("system", sys), ("human", "{q}")])
    msg = tmpl.format_messages(q=prompt_text)
    return invoke_with_fallback(llm, msg, groq_key).content.strip()

def build_brief(query: str, docs: List[Document], ln: str) -> str:
    llm = make_llm("llama-3.1-8b-instant", groq_key)
    src_blob = format_sources_for_prompt(docs)
    tmpl = ChatPromptTemplate.from_messages([("system", SYSTEM_SUMMARY), ("human", HUMAN_SUMMARY)])
    msg = tmpl.format_messages(query=query, sources=src_blob, lang_name=ln)
    return invoke_with_fallback(llm, msg, groq_key).content.strip()

def answer_from_brief(query: str, brief: str, ln: str) -> str:
    llm = make_llm("llama-3.1-8b-instant", groq_key)
    tmpl = ChatPromptTemplate.from_messages([("system", SYSTEM_QA), ("human", HUMAN_QA)])
    msg = tmpl.format_messages(query=query, brief=brief, lang_name=ln)
    return invoke_with_fallback(llm, msg, groq_key).content.strip()

def build_citations(docs: List[Document]) -> List[Tuple[str, List[int]]]:
    collated: Dict[str, set] = {}
    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", None)
        if src not in collated:
            collated[src] = set()
        if isinstance(page, int):
            collated[src].add(page)
    return [(fn, sorted(pages)) for fn, pages in collated.items()]

# ---------------- Chat History --------------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ---------------- Chat Input ----------------
prompt = st.chat_input("Type here… (prefix with /chat to bypass RAG)")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        user_lang = detect_lang(prompt)
        ln = lang_name(user_lang)

        force_chat = prompt.strip().lower().startswith("/chat")
        have_index = st.session_state.vs is not None
        force_rag = wants_docs_explicitly(prompt)

        # 1) Plain chat if no index or user forced /chat
        if not have_index or force_chat:
            reply = llm_reply(prompt[5:].strip() if force_chat else prompt)
            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.stop()

        # 2) RAG path (auto-routed; but forced if user asked for PDFs/sources)
        summary_mode = detect_summary_intent(prompt)
        if summary_mode and st.session_state.all_chunks:
            docs = sample_evenly(st.session_state.all_chunks, per_file=10)
        else:
            if use_mmr and hasattr(st.session_state.vs, "max_marginal_relevance_search"):
                docs = st.session_state.vs.max_marginal_relevance_search(prompt, k=TOP_K, fetch_k=max(4*TOP_K, 20))
            else:
                docs = st.session_state.vs.similarity_search(prompt, k=TOP_K)

            # Router: only fallback to chat when user didn't ask for docs explicitly
            if (not force_rag) and keyword_overlap_router(prompt, docs):
                reply = llm_reply(prompt)
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
                st.stop()

        # 3) Build brief -> QA (always from docs here)
        brief = build_brief(prompt, docs, ln)
        answer = answer_from_brief(prompt, brief, ln)

        # If QA is weak, fall back to general chat ONLY if user didn't force RAG
        if (not force_rag) and looks_like_nonanswer(answer):
            answer = llm_reply(prompt)
            cites = []
        else:
            cites = build_citations(docs)

        # 4) Render (answer + citations)
        response = f"**Answer**\n{answer}\n\n"
        if cites:
            response += "**Citations**\n"
            for fn, pages in sorted(cites, key=lambda x: x[0].lower()):
                pages_str = ", ".join(map(str, pages)) if pages else "?"
                response += f"- {fn} (pages: {pages_str})\n"

        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
