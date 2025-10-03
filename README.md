🤖 Multi-Agent RAG (Groq + LangChain + Streamlit)

A single-file Streamlit app (app.py) that:

accepts user questions about one or more PDFs,

summarizes when asked,

answers via a Retriever → Summarizer → QA → Citations pipeline,

works in English and Bengali (বাংলা) end-to-end,

and gracefully falls back to normal chat when a question isn’t about your PDFs.

✨ Features

Auto-indexing: upload PDFs → vectors are built automatically (FAISS).

Multi-PDF: drag & drop multiple files; questions can span all of them.

Bilingual: multilingual embeddings + language detection → answers in the user’s language (English/Bengali).

Two modes (auto)

Summary when you ask “summarize / সারসংক্ষেপ …”

RAG QA otherwise (retrieves relevant chunks, then answers from a brief)

Force RAG w/ citations: phrases like “from the pdf / cite / উৎস / পৃষ্ঠা” guarantee answers grounded in the docs (file + page).

Router: if a question clearly doesn’t match the PDFs, it falls back to general chat (avoids fake citations).

MMR retrieval (toggle) to reduce duplicate chunks.

Model fallback: if a Groq model is deprecated, the app auto-switches to a supported one.

🧰 Stack

UI: Streamlit

LLM: Groq (llama-3.1-8b-instant, fallback to llama-3.3-70b-versatile)

RAG: LangChain (FAISS + multilingual sentence transformers)

Embeddings: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

PDF parsing: PyPDFLoader (swap to PyMuPDF loader if a Bengali PDF extracts poorly)

📁 Project layout
.
├─ app.py           # the entire app
├─ requirements.txt # Python deps
└─ .env             # your GROQ_API_KEY (not committed)

✅ Prerequisites

Python 3.9–3.11

A Groq API key: https://console.groq.com/keys

🚀 Quick start
# 1) create and activate a venv
python -m venv venv
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 2) install deps
pip install -r requirements.txt

# 3) add your key (either .env or paste in the app prompt)
echo GROQ_API_KEY=your_key_here > .env

# 4) run
streamlit run app.py


When the app opens, you can also paste your Groq key if .env is missing.

🔐 .env
GROQ_API_KEY=your_groq_api_key_here


The app prompts for a key if none is found.

🧑‍💻 How to use

Upload PDFs in the sidebar. Indexing happens automatically.

Ask in English or Bengali:

“summarize the pdf” / “সারসংক্ষেপ দাও” → Summary

“Explain Bellman–Ford” / “বিষয়ের সময়জটিলতা কত?” → RAG QA

To force citations and use the PDFs no matter what, include cues like:

“from the pdf”, “cite”, “source”

“পিডিএফ থেকে”, “উৎস দিন”, “রেফারেন্স”, “পৃষ্ঠা”

To bypass documents and chat normally even with PDFs loaded, start your message with /chat.

Use Clear Index to wipe vectors from memory.

🧠 Behavior details
Language

The app detects the message language and instructs the LLM to respond in English or Bengali.

Multilingual embeddings allow English↔Bengali retrieval across mixed-language docs.

Summary vs QA

Summary triggers (EN + BN) include:
summarize, summary, overview, tl;dr, সারসংক্ষেপ, সার, সংক্ষেপে, সারাংশ, পিডিএফটা কি সম্পর্কে …
Summary mode samples chunks evenly across each PDF so the brief covers the whole doc.

Otherwise, the app runs Retriever → Summarizer → QA and returns Answer + Citations.

Router & Force RAG

If your question doesn’t match the PDFs, the app answers in plain chat (no citations) to prevent hallucinated references.

If you explicitly ask “from the pdf / cite / উৎস”, the router is skipped and citations are guaranteed.

⚙️ Tuning knobs

Sidebar controls:

Use MMR (diverse retrieval) – promotes variety in retrieved chunks (on by default).

Upload PDFs – supports multiple files.

Clear Index – resets vectors.

Defaults (tuned for ~8 GB RAM):

CHUNK_SIZE = 1600
CHUNK_OVERLAP = 200
TOP_K = 8


You can change them in app.py.

📦 requirements.txt
streamlit
python-dotenv
langchain
langchain-community
langchain-text-splitters
langchain-groq
sentence-transformers
faiss-cpu
pypdf
langdetect
# optional for better Unicode extraction in some BN PDFs:
# pymupdf


If some Bengali PDFs show garbled text, add pymupdf and switch to the PyMuPDF loader (see Troubleshooting).



🧩 Architecture mapping
User Query
   │
   ▼
Supervisor (router: summary vs RAG vs chat)
   │
   ├── Retriever Agent (FAISS, optional MMR)
   │
   ├── Summarizer Agent (builds grounded brief)
   │
   ├── QA Agent (answers only from brief)
   │
   └── Citation Agent (file + page)
        ▼
Final Answer (+ Citations)

🔭 Ideas to extend

Persistent index (Chroma) to keep vectors across app restarts.

OCR fallback for scanned PDFs (pdf2image + pytesseract).

Inline citation markers like [1] in the answer body.

Deployment on Streamlit Community Cloud, Fly.io, Render, or a small VM.

📜 License

Add a license of your choice (e.g., MIT) at the repo root as LICENSE.