# Project Hilbert / 希尔伯特计划

Project Hilbert is a terminal-first concept vector exploration tool. It maps words and concepts into a vector space, searches nearby concepts, performs simple vector logic, clusters neighboring results, and estimates a `salient` value for each candidate.

Project Hilbert 是一个面向终端（terminal-first）的概念向量空间探索工具：把词语与概念映射到向量空间中，进行近邻搜索、向量逻辑运算、结果聚类，并为候选项估计 `salient`（显著性）值。

The default model for v0.1 is an offline deterministic semantic mock embedding. This keeps the project runnable without downloading large models. Optional real embedding support is prepared for models such as `BAAI/bge-m3`.

v0.1 的默认模型是离线、确定性的语义 mock embedding，确保无需下载大模型也能运行；同时也预留了真实 embedding（如 `BAAI/bge-m3`）的可选支持。

For real exploration, Project Hilbert should not rely on the tiny demo file `data/concepts.zh.txt`. Use a large public Chinese lexicon and build a persistent Chroma collection, then query that collection.

若要进行真实探索，不应依赖小型 demo 文件 `data/concepts.zh.txt`；建议使用公开中文大词表并构建持久化的 Chroma collection，然后基于该 collection 进行查询。

## Documentation / 文档

- `fiat_lux.md`: Vision whitepaper / 愿景白皮书
- `build_plan.md`: Engineering plan & design notes / 工程计划与设计说明
- `corpus/README.md`: Corpus directory notes / 词表目录说明

## Features / 功能

- Nearest-neighbor concept search / 概念近邻搜索
- Vector logic expressions (`A + B`, `A - B + C`) / 向量逻辑表达式（`A + B`, `A - B + C`）
- Optional DBSCAN clustering of results / 可选的 DBSCAN 结果聚类
- `salient` score for “interestingness” / `salient` 显著性评分
- Output formats: table, JSON, Markdown / 输出格式：表格、JSON、Markdown

## Requirements / 运行环境

- Python 3.10+ (recommended) / 建议 Python 3.10+
- No third-party dependencies for the default `--model mock` mode / 默认 `--model mock` 模式零第三方依赖

## Installation / 安装

Default mode (no dependencies) / 默认模式（无依赖）:

```bash
python main.py --help
```

Optional dependencies (real embeddings + Chroma) / 可选依赖（真实 embedding + Chroma）:

```bash
python -m pip install -r requirements-optional.txt
```

## Quick start / 快速开始

From the project root / 在项目根目录执行：

```bash
python main.py --help
python main.py models
python main.py search "苹果" --top-n 10
python main.py logic "北京 - 中国 + 英国" --top-n 10
```

JSON output / JSON 输出：

```bash
python main.py search "苹果" --format json
```

Markdown output / Markdown 输出：

```bash
python main.py logic "国王 - 男人 + 女人" --format markdown
```

## Commands / 命令

```text
search <query>      Search concepts near a query. / 概念近邻搜索
logic <expression>  Evaluate + / - vector logic and search nearest concepts. / 向量逻辑表达式
import-jieba        Import jieba's public Chinese dictionary. / 导入 jieba 公开词典
import-words        Clean and merge public/user word files. / 清洗合并词表文件
build-chroma        Embed a lexicon and persist it into Chroma. / 构建并持久化 Chroma collection
models              Show supported model aliases. / 显示可用模型别名
```

## Common options / 常用参数

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

## Large Chinese lexicon + Chroma / 大型中文词表 + Chroma

Embedding models map text to vectors; they do not directly decode arbitrary vectors back to words. Project Hilbert maps vectors back to words by searching a large Chroma collection of candidate terms.

Embedding 模型做的是 text -> vector；它不能把任意向量直接“解码回词语”。Project Hilbert 通过在 Chroma 中对大规模候选词表做最近邻检索，从向量回到词语候选列表。

Start with jieba's public Chinese dictionary / 从 jieba 的公开词典开始：

```bash
python -m pip install "chromadb>=0.5.0" "jieba>=0.42.1"
python main.py import-jieba --out corpus/jieba_terms.txt
python main.py build-chroma --model bge-m3 --concepts corpus/jieba_terms.txt --collection project_hilbert_zh --persist-dir ./.chroma --batch-size 64 --reset
```

Search the Chroma collection instead of the demo list / 查询 Chroma collection（而不是 demo 小词表）：

```bash
python main.py search "苹果" --model bge-m3 --collection project_hilbert_zh --persist-dir ./.chroma --top-n 30 --search-k 200
python main.py logic "北京 - 中国 + 英国" --model bge-m3 --collection project_hilbert_zh --persist-dir ./.chroma --top-n 30 --search-k 200
```

Merge other public word files, such as THUOCL, Wikipedia titles, Wikidata labels, OpenHowNet terms, or your own vocabulary files / 合并更多公开词表（如 THUOCL、Wikipedia 标题、Wikidata labels、OpenHowNet、以及你的自定义词表）：

```bash
python main.py import-words "./corpus/raw/thuocl.txt" "./corpus/raw/wiki_titles.txt" --out "./corpus/zh_terms.txt"
python main.py build-chroma --model bge-m3 --concepts "./corpus/zh_terms.txt" --collection project_hilbert_zh --persist-dir "./.chroma" --batch-size 64 --reset
```

## Model modes / 模型模式

### `mock` (default) / `mock`（默认）

```bash
python main.py search "苹果" --model mock
```

- No dependency installation required. / 无需安装依赖
- No network required. / 无需联网
- Good for demos, tests, and first-run terminal IO. / 适合演示、测试与首次运行
- Uses hand-designed semantic features plus deterministic hash noise. / 使用手工语义特征 + 确定性哈希噪声

### `mini` (optional) / `mini`（可选）

```bash
python -m pip install "sentence-transformers>=3.0.0"
python main.py search "苹果" --model mini
```

Loads / 加载：

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

### `bge-m3` (optional) / `bge-m3`（可选）

```bash
python -m pip install "sentence-transformers>=3.0.0"
python main.py search "苹果" --model bge-m3
```

Loads / 加载：

```text
BAAI/bge-m3
```

The first run downloads the model to the local Hugging Face cache. Later runs can reuse the local cache.

首次运行会把模型下载到本机 Hugging Face cache；后续运行可复用缓存。

If `models/bge-m3` exists in the project root, `--model bge-m3` automatically loads that local directory first / 如果项目根目录存在 `models/bge-m3`，则会优先加载本地目录：

```bash
python main.py search "苹果" --model bge-m3 --top-n 10
python main.py logic "北京 - 中国 + 英国" --model bge-m3 --top-n 10
python main.py logic "国王 - 男人 + 女人" --model bge-m3 --format json
```

You can also pass a local path explicitly / 也可以显式传入本地路径：

```bash
python main.py search "苹果" --model "./models/bge-m3" --top-n 10
```

## Examples / 示例

Search / 近邻搜索：

```bash
python main.py search "苹果" --top-n 8
```

Vector logic / 向量逻辑：

```bash
python main.py logic "北京 - 中国 + 英国" --top-n 8
python main.py logic "国王 - 男人 + 女人" --top-n 8
```

Use a custom concept file / 使用自定义概念文件：

```bash
python main.py --concepts data/concepts.zh.txt search "人工智能"
```

## Run tests / 运行测试

```bash
python -m unittest discover -s tests
```

## Project layout / 项目结构

```text
project_hilbert/
  embeddings.py      Embedding adapters: mock and optional sentence-transformers. / embedding 适配器
  index.py           In-memory vector index. / 内存向量索引
  chroma_store.py    Persistent Chroma vector store for large lexicons. / Chroma 持久化存储
  corpus.py          Public/user lexicon import and cleaning utilities. / 词表导入与清洗
  clustering.py      Dependency-free DBSCAN implementation. / 纯 Python DBSCAN
  salience.py        Salient value estimation. / 显著性估计
  vector_logic.py    + / - expression parser and evaluator. / 向量逻辑解析与执行
  engine.py          Query orchestration. / 查询编排
  renderers.py       Table / JSON / Markdown output. / 输出渲染
  cli.py             Terminal interface. / CLI 接口
data/
  concepts.zh.txt    Small demo seed concept library only. / 小型 demo 概念库
corpus/
  *.txt              Large public/user lexicons for Chroma indexing. / 大词表（通常不入库）
```

## Current limitations / 当前限制

- The default `mock` embedding is designed for engineering validation and demos, not scientific measurement. / 默认 `mock` 用于工程验证与演示，不代表科学测量
- `data/concepts.zh.txt` is only a small demo list; real exploration should use Chroma with a large lexicon. / demo 小词表不适用于真实探索，建议使用 Chroma + 大词表
- Real semantic exploration should use `BAAI/bge-m3`, MiniLM, E5, BGE, GTE, Jina, or another trained embedding model. / 真实语义探索应使用训练过的 embedding 模型
- DBSCAN parameters may need adjustment for real embedding spaces. / DBSCAN 参数在真实空间中需要调参
- Vector arithmetic is approximate and model-dependent. / 向量算术是近似的，且与模型相关

## Fiat Lux / 愿景

Project Hilbert is an attempt to make concepts visible as geometry: nearby terms, hidden clusters, analogical directions, sparse bridges, and surprising relations.

Project Hilbert 尝试把概念作为“几何对象”呈现出来：近邻、隐藏簇、类比方向、稀疏桥梁，以及那些令人意外的联系。更多背景与动机见 `fiat_lux.md`。
