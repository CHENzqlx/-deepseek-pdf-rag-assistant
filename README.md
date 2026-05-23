# DeepSeek PDF RAG Assistant

A Streamlit-based PDF question answering app using RAG. Upload one or more PDF files, build a local FAISS vector index, and ask questions about the document content with DeepSeek.

## Features

- Upload and process multiple PDF files
- Extract PDF text with PyPDF2
- Split long documents into searchable chunks
- Store document vectors locally with FAISS
- Answer questions using DeepSeek's OpenAI-compatible API
- Keep API keys out of Git with `.env`

## Setup

1. Create and activate a virtual environment:

```sh
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```sh
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

3. Create a `.env` file:

```env
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

4. Run the app:

```sh
streamlit run app.py
```

## Usage

1. Open `http://localhost:8501`.
2. Upload PDF files in the sidebar.
3. Click `Submit & Process` to create the FAISS index.
4. Ask questions in the input box.

## Project Structure

```text
.
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```
