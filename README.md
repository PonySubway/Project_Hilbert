# Project Hilbert

Project Hilbert is a terminal-first concept vector exploration tool. It maps words and concepts into a vector space, searches nearby concepts, performs simple vector logic, clusters neighboring results, and estimates a `salient` value for each candidate.

For the first public version, the default model is an offline deterministic semantic mock embedding. This keeps the project runnable without downloading large models. Optional real embedding support is prepared for models such as `BAAI/bge-m3`.

For real exploration, Project Hilbert should not rely on the tiny demo file `data/concepts.zh.txt`. Use a large public Chinese lexicon and build a persistent Chroma collection, then query that collection.

## Quick start

From the project root:

```powershell
python main.py --help
python main.py models
python main.py search "苹果" --top-n 10
python main.py logic "北京 - 中国 + 英国" --top-n 10
```

JSON output:

```powershell
python main.py search "苹果" --format json
```

Markdown output:

```powershell
python main.py logic "国王 - 男人 + 女人" --format markdown
```

## Commands

```text
search <query>      Search concepts near a query.
logic <expression>  Evaluate + / - vector logic and search nearest concepts.
import-jieba        Import jieba's public Chinese dictionary.
import-words        Clean and merge public/user word files.
build-chroma        Embed a lexicon and persist it into Chroma.
models              Show supported model aliases.
```

Common options:

```text
--model mock|mini|bge-m3|<sentence-transformers-name>
--concepts data/concepts.zh.txt
--collection project_hilbert_zh
--persist-dir .chroma
--top-n 12
--format table|json|markdown
--no-cluster
--eps 0.35
--min-samples 2
```

## Large Chinese lexicon + Chroma

Embedding models map text to vectors; they do not directly decode arbitrary vectors back to words. Project Hilbert maps vectors back to words by searching a large Chroma collection of candidate terms.

Start with jieba's public Chinese dictionary:

```powershell
pip install "chromadb>=0.5.0" "jieba>=0.42.1"
python main.py import-jieba --out ".\corpus\jieba_terms.txt"
python main.py build-chroma --model bge-m3 --concepts ".\corpus\jieba_terms.txt" --collection project_hilbert_zh --persist-dir ".\.chroma" --batch-size 64 --reset
```

Search the Chroma collection instead of the demo list:

```powershell
python main.py search "苹果" --model bge-m3 --collection project_hilbert_zh --persist-dir ".\.chroma" --top-n 30 --search-k 200
python main.py logic "北京 - 中国 + 英国" --model bge-m3 --collection project_hilbert_zh --persist-dir ".\.chroma" --top-n 30 --search-k 200
```

Merge other public word files, such as THUOCL, Wikipedia titles, Wikidata labels, OpenHowNet terms, or your own vocabulary files:

```powershell
python main.py import-words ".\corpus\raw\thuocl.txt" ".\corpus\raw\wiki_titles.txt" --out ".\corpus\zh_terms.txt"
python main.py build-chroma --model bge-m3 --concepts ".\corpus\zh_terms.txt" --collection project_hilbert_zh --persist-dir ".\.chroma" --batch-size 64 --reset
```

## Model modes

### `mock` 默认模式

```powershell
python main.py search "苹果" --model mock
```

- No dependency installation required.
- No network required.
- Good for demos, tests, and first-run terminal IO.
- Uses hand-designed semantic features plus deterministic hash noise.

### `mini` optional mode

```powershell
pip install "sentence-transformers>=3.0.0"
python main.py search "苹果" --model mini
```

Loads:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

### `bge-m3` optional mode

```powershell
pip install "sentence-transformers>=3.0.0"
python main.py search "苹果" --model bge-m3
```

Loads:

```text
BAAI/bge-m3
```

The first run downloads the model to the local Hugging Face cache. Later runs can reuse the local cache.

If `models/bge-m3` exists in the project root, `--model bge-m3` automatically loads that local directory first:

```powershell
python main.py search "苹果" --model bge-m3 --top-n 10
python main.py logic "北京 - 中国 + 英国" --model bge-m3 --top-n 10
python main.py logic "国王 - 男人 + 女人" --model bge-m3 --format json
```

You can also pass the local path explicitly:

```powershell
python main.py search "苹果" --model ".\models\bge-m3" --top-n 10
```

## Examples

Search:

```powershell
python main.py search "苹果" --top-n 8
```

Vector logic:

```powershell
python main.py logic "北京 - 中国 + 英国" --top-n 8
python main.py logic "国王 - 男人 + 女人" --top-n 8
```

Use a custom concept file:

```powershell
python main.py --concepts data/concepts.zh.txt search "人工智能"
```

## Run tests

```powershell
python -m unittest discover -s tests
```

## Project files

```text
project_hilbert/
  embeddings.py      Embedding adapters: mock and optional sentence-transformers.
  index.py           In-memory vector index.
  chroma_store.py    Persistent Chroma vector store for large lexicons.
  corpus.py          Public/user lexicon import and cleaning utilities.
  clustering.py      Dependency-free DBSCAN implementation.
  salience.py        Salient value estimation.
  vector_logic.py    + / - expression parser and evaluator.
  engine.py          Query orchestration.
  renderers.py       Table / JSON / Markdown output.
  cli.py             Terminal interface.
data/
  concepts.zh.txt    Small demo seed concept library only.
corpus/
  *.txt              Large public/user lexicons for Chroma indexing.
```

## Current limitations

- The default `mock` embedding is designed for engineering validation and demos, not scientific measurement.
- `data/concepts.zh.txt` is only a small demo list; real exploration should use Chroma with a large lexicon.
- Real semantic exploration should use `BAAI/bge-m3`, MiniLM, E5, BGE, GTE, Jina, or another trained embedding model.
- DBSCAN parameters may need adjustment for real embedding spaces.
- Vector arithmetic is approximate and model-dependent.

## Fiat Lux

Project Hilbert is an attempt to make concepts visible as geometry: nearby terms, hidden clusters, analogical directions, sparse bridges, and surprising relations.

