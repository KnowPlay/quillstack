<!-- PROJECT SUMMARY -->
<div align="center">
  <h1 align="center">QuillStack</h1>

  <p align="center">
    An interactive Streamlit RAG chatbot for uploading and querying PDFs
    <br>
    <a href="https://github.com/KnowPlay/quillstack/issues">» submit a suggestion </a>
    ·
    <a href="https://github.com/KnowPlay/quillstack/issues">» report a bug </a>
    ·
    <a href="https://github.com/KnowPlay/quillstack">» contact </a>
  </p>

  <div align="center">

![GitHub forks](https://img.shields.io/github/forks/KnowPlay/quillstack?style=social) ![GitHub stars](https://img.shields.io/github/stars/KnowPlay/quillstack?style=social)

<!-- [![CI](https://github.com/org-name/repo-name/actions/workflows/file-name.yml/badge.svg)](https://github.com/KnowPlay/org-name/repo-name/actions/workflows/file-name.yml) -->

[![CI](https://github.com/KnowPlay/quillstack/actions/workflows/push_on_main.yml/badge.svg)](https://github.com/KnowPlay/quillstack/actions/workflows/push_on_main.yml)
![GitHub Pull Request (open)](https://img.shields.io/github/issues-pr/KnowPlay/quillstack?color=blue) ![GitHub last commit](https://img.shields.io/github/last-commit/KnowPlay/quillstack?color=pink) ![GitHub License](https://img.shields.io/github/license/KnowPlay/quillstack?color=green) ![contributions welcome](https://img.shields.io/badge/contributions-welcome-purple.svg?style=flat)

  </div>
</div>

<!-- TABLE OF CONTENT -->
<details open="open">
  <summary><h2 style="display: inline-block">🕹 Table of Content</h2></summary>
  <ol>
    <li>
      <a href="#🌻-about">About</a>
      <ul>
        <li><a href="#🔧-tech-stack">Tech Stack</a></li>
        <li><a href="#🍄-features">Features</a></li>
      </ul>
    </li>
    <li>
      <a href="#🌵-documentation">Documentation</a>
      <ul>
        <li><a href="#🍯-setup">Setup</a></li>
        <li><a href="#🍎-development">Development</a></li>
      </ul>
    </li>
    <li><a href="#🌾-contributing">Contributing</a></li>
    <li><a href="#📜-license">License</a></li>
  </ol>
</details>

## 🌻 About

QuillStack is a Streamlit-based application enabling you to upload, index, and conversationally query multiple PDF documents using Retrieval-Augmented Generation (RAG). It combines LangChain, FAISS, and OpenAI/Anthropic LLMs to let you chat with your PDFs.

### 🔧 Tech Stack

#### ➕ Development Tools

- [x] Python 3.9+

#### ➕ Backend

- [x] LangChain for RAG workflows
- [x] PyPDF2 for PDF parsing
- [x] spaCy (`en_core_web_sm`) for embeddings
- [x] FAISS for vector similarity search
- [x] OpenAI / Anthropic Python SDKs

#### ➕ Frontend

- [x] Streamlit for interactive UI

#### ➕ DevOps

- [ ] GitHub Actions (CI via `push_on_main.yml`)
- [ ] Deployment (Streamlit Cloud / Heroku)

### 🍄 Features

#### ➕ Multi-PDF Upload & Indexing

- [ ] Upload multiple PDF files
- [ ] Extract and concatenate text from pages
- [ ] Split text into chunks
- [ ] Persist a local FAISS vector index

#### ➕ Conversational Chat

- [ ] Enter free-form questions
- [ ] Retrieve relevant context via vector search
- [ ] Generate answers with LLMs
- [ ] Fallback: “answer is not available in the context”

<!-- CONTENT -->

## 🌵 Documentation

### 🍯 Setup

- [x] **Clone the repository**
  ```bash
  git clone quillstack.git
  cd quillstack
  ```
- [x] **Create & activate a virtual environment**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```
- [x] **Install dependencies**
  ```bash
  pip install -r requirements.txt
  python -m spacy download en_core_web_sm
  ```
- [x] **Configure environment variables**  
       Create a `.env` file:
  ```dotenv
  OPENAI_API_KEY=sk-...
  ANTHROPIC_API_KEY=...  # optional
  ```
- [x] **Run the application**
  ```bash
  streamlit run app.py
  ```
- [x] **Open** http://localhost:8501 your browser.

### 🍎 Development

- [ ] Add new LLM providers
- [ ] Customize chunking parameters (`chunk_size`, `chunk_overlap`)
- [ ] Enhance the UI (authentication, file management, highlighting)
- [ ] Extend CI workflows in `.github/workflows`
- [ ] Write unit tests for PDF ingestion & retrieval

<!-- CONTRIBUTING -->

## :ear_of_rice: Contributing

<!-- Add contribution guidelines here -->

> 1. Fork the Project
> 2. Create your Branch (`git checkout -b my-branch`)
> 3. Commit your Changes (`git commit -m 'add my contribution'`)
> 4. Push to the Branch (`git push --set-upstream origin my-branch`)
> 5. Open a Pull Request

<!-- LICENSE -->

## :pencil: License

<!-- Add license information here -->

This project is licensed under [MIT](https://opensource.org/licenses).

<!-- ACKNOWLEDGEMENTS -->
<!-- ## Acknowledgements -->
