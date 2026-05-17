# devColorBot

A one-command FAQ chatbot for [/dev/color](https://devcolor.org). Ask questions; answers are grounded in the official FAQ using RAG and a local AI.

## Run

Pick **one** of these — you do **not** need `pip install -e .`, `source .venv`, or `.venv/bin` on your `PATH` manually. The [`devcolorbot`](devcolorbot) launcher creates the virtual environment and installs dependencies for you.

### Option 1 — From the project folder (zero setup)

```bash
git clone https://github.com/ChrxsCrxss/devcolor-chat-bot.git
cd devcolor-chat-bot
./devcolorbot
```

Requires Python 3.11+ on your Mac. First run may take a few minutes (deps, Ollama, model, index).

### Option 2 — `devcolorbot` from anywhere (one-time PATH)

Add the project folder to your shell profile **once** (not `.venv/bin`):

```bash
echo 'export PATH="$HOME/Projects/devcolor-chat-bot:$PATH"' >> ~/.zshrc
source ~/.zshrc
devcolorbot
```

Adjust the path to wherever you cloned the repo.

### Option 3 — Global install with pipx

```bash
git clone https://github.com/ChrxsCrxss/devcolor-chat-bot.git
cd devcolor-chat-bot
pipx install .
devcolorbot
```

`pipx` puts `devcolorbot` in `~/.local/bin` (usually already on `PATH`).

---

**First run** automatically:

1. Creates `.venv` and installs Python dependencies  
2. **Installs Ollama** if missing (macOS/Linux: [install.sh](https://ollama.com/install.sh); Windows: `winget install Ollama.Ollama`), starts the API, and downloads the model (`llama3.2:3b`)  
3. Builds the FAQ vector index (cached under `.cache/index/`)  
4. Opens the chat  

If setup fails, the bot exits with instructions — run `devcolorbot setup` to retry. Use `--echo` only for an offline FAQ demo without Ollama.

**Later runs** start the chat right away (index and model are cached).

**Note:** If you moved or renamed the repo, delete `.venv` and run `./devcolorbot` again so the virtualenv is recreated with correct paths.

## What you’ll see

```
devColorBot · /dev/color FAQ
...
> How can /dev/color help my career?

devColorBot > ✨ ...
```

Type `help`, `/SET`, or `exit`.

## Optional flags (power users)

| Flag | Purpose |
|------|---------|
| `--doctor` | Check Python, RAM, Ollama, and cache |
| `--once "question"` | Single answer, then exit |
| `--profile light` | Smaller model for low-RAM Macs |
| `--echo` | Offline FAQ demo without Ollama (no install attempted) |
| `--skip-setup` | Skip Ollama install (fails at chat if Ollama is not already running) |
| `devcolorbot setup` | Install/repair Ollama and pull the model |
| `--debug` | Extra retrieval detail |

## About /dev/color

/dev/color is a global career accelerator for Black software engineers and technologists — community, mentorship, programs like the **A\*** Program, and industry partnerships.

## What is RAG?

1. Chunk the FAQ  
2. Embed chunks as vectors  
3. Retrieve relevant chunks for your question  
4. Augment the AI prompt with that context  
5. Generate an answer grounded in the corpus  

## Technologies

| Layer | Technology | Role |
|-------|------------|------|
| Language | Python 3.11+ | Application runtime |
| Packaging | `venv`, `pip`, `setuptools` | Dependency install; editable package layout |
| Embeddings | [sentence-transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`) | Semantic vectors for FAQ retrieval (`balanced` / `quality` profiles) |
| Embeddings (light) | [scikit-learn](https://scikit-learn.org/) TF-IDF | Lexical fallback for low-RAM machines |
| Vector search | [NumPy](https://numpy.org/) | In-memory cosine similarity (no external vector DB) |
| LLM | [Ollama](https://ollama.com) | Local inference (`llama3.2:3b` default; profiles swap model size) |
| LLM API | `requests` | HTTP client for Ollama chat API |
| CLI / UX | [Rich](https://rich.readthedocs.io/) | Terminal styling, tables, progress spinners |
| System checks | `psutil` | RAM hints in `doctor` |
| Corpus | Plain-text FAQ (`data/devcolorfaq.txt`) | Source knowledge base |
| Cache | JSON + `.npy` on disk (`.cache/index/`) | Persisted embedding index between runs |
| Observability | `.log` + `.json` per turn (`logs/`) | Human-readable and structured RAG traces |

**Architecture pattern:** Retrieval-Augmented Generation (RAG) — retrieve relevant FAQ context, augment the prompt, generate with a local LLM. No cloud API keys required.

## Design

| Piece | Choice |
|-------|--------|
| Chunking | Sentence-level retrieval; full Q&A in the prompt |
| Embeddings | MiniLM (`balanced`) or TF-IDF (`light`) |
| LLM | Ollama (`llama3.2:3b` default) |
| CLI | `devColorBot >` / user lines with `< user` |
| Index | Cached in `.cache/index/` |

## Hardware

| Profile | RAM | Model |
|---------|-----|-------|
| `light` | 4–8 GB | `llama3.2:1b` |
| `balanced` (default) | 8+ GB | `llama3.2:3b` |
| `quality` | 16 GB | `phi3:mini` |

First run downloads ~2GB for the Ollama model plus embedding dependencies.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `command not found: devcolorbot` | Use `./devcolorbot` from the project folder, add the **project folder** to `PATH` (see Option 2), or `pipx install .` |
| `bad interpreter` / venv errors after moving the repo | `rm -rf .venv` then `./devcolorbot` (recreates the venv) |
| `permission denied` | `chmod +x devcolorbot` then run again |
| Stuck on Ollama install | Run: `OLLAMA_NO_START=1 curl -fsSL https://ollama.com/install.sh \| sh`, then `devcolorbot` again |
| Demo-style answers | Ollama not ready — run `devcolorbot` again (starts `ollama serve` in the background) |
| Full environment check | `devcolorbot --doctor` |

## Sample output

See [`sample_output.txt`](sample_output.txt) for the three assignment demo questions. Regenerate after Ollama is running:

```bash
bash scripts/capture_demo.sh
```

## Project layout

```
devcolor-chat-bot/
  devcolorbot          # launcher (bootstraps .venv on first run)
  data/devcolorfaq.txt # FAQ corpus (required)
  src/devcolor_rag/
  .cache/index/        # embedding cache (auto-created, gitignored)
  .venv/               # auto-created; do not commit
```

## License

Take-home assessment project.
