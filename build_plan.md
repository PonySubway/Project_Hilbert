# Project Hilbert Build Plan

> 本文件面向智能体与工程协作者。  
> 目标：把 `fiat_lux.md` 中的愿景落地为一个可运行、可测试、可扩展的 Python 项目。

## 0. v0.1 实现状态

当前第一版已经实现为本地终端 IO 工具，暂不包含 GUI。

已完成：

- `python main.py --help` CLI 入口；
- `search` 概念近邻搜索；
- `logic` 向量逻辑表达式搜索；
- `models` 模型说明命令；
- 零外部依赖的 `semantic-mock` embedding，便于离线演示与测试；
- 可选 `sentence-transformers` 适配器，支持后续加载 `BAAI/bge-m3`、MiniLM 或其他模型；
- 纯 Python 内存向量索引；
- 纯 Python DBSCAN 聚类；
- salient value 显著性评分；
- table / JSON / Markdown 输出；
- `data/concepts.zh.txt` 初始 demo 概念库；
- 本地 `BAAI/bge-m3` 加载；
- Chroma 大词表索引路线：`import-jieba`、`import-words`、`build-chroma`、`search --collection`、`logic --collection`；
- `unittest` 测试套件。

当前验证命令：

```powershell
python -m unittest discover -s tests
python main.py search "苹果" --top-n 10
python main.py logic "北京 - 中国 + 英国" --top-n 8
python main.py logic "国王 - 男人 + 女人" --format json
python main.py import-jieba --out corpus/jieba_terms.txt
python main.py build-chroma --model bge-m3 --concepts corpus/jieba_terms.txt --collection project_hilbert_zh --reset
python main.py search "苹果" --model bge-m3 --collection project_hilbert_zh --top-n 30 --search-k 200
```

## 1. 项目目标

Project Hilbert 是一个概念向量空间探索工具。它使用开源 embedding 模型，将语词、短语、句子映射到向量空间中，并提供以下能力：

1. 概念近邻搜索；
2. 自动语义聚类；
3. 向量逻辑运算；
4. salient value 显著性评分；
5. 人类可读与机器可读的结果输出。

## 2. MVP 范围

第一版只实现本地命令行工具，不做 Web UI。当前方向从“小型概念列表 demo”升级为“公开中文大词表 + Chroma 向量数据库”。

### 必须实现

- 从文本概念列表构建向量索引；
- 从公开中文词典 / 百科标题 / 用户词表构建大型词汇宇宙；
- 使用 Chroma 持久化 embedding 与 metadata；
- 输入一个查询词，返回相近概念；
- 使用 DBSCAN 对近邻结果自动分簇；
- 如果聚类效果不明显，允许输出未分簇结果；
- 为每个结果输出：
  - `concept`
  - `similarity`
  - `distance`
  - `cluster_id`
  - `salient`
- 支持基本向量表达式：
  - `A + B`
  - `A - B`
  - `A - B + C`
- 支持 JSON 与 Markdown 输出。

### 暂不实现

- Web 前端；
- 大规模分布式索引；
- 用户登录；
- 复杂知识图谱；
- 严格数学意义上的语义证明系统。

## 3. 推荐技术栈

语言：Python 3.10+

核心依赖建议：

```text
numpy
scikit-learn
sentence-transformers
chromadb
jieba
rich
pydantic
typer
```

可选依赖：

```text
hdbscan
umap-learn
plotly
faiss-cpu
```

说明：

- `sentence-transformers`：加载开源 embedding 模型；
- `chromadb`：本地持久向量数据库，负责大词表 ANN 检索与 metadata 存储；
- `jieba`：提供公开中文词典作为第一批大规模词汇来源；
- `numpy`：向量运算；
- `scikit-learn`：cosine similarity、DBSCAN、NearestNeighbors；
- `typer`：命令行接口；
- `rich`：美化终端输出；
- `pydantic`：规范数据结构。

## 3.1 公开中文词汇来源

Project Hilbert 不应依赖 `data/concepts.zh.txt` 这种小型手写列表。该文件只保留为 smoke test / demo seed。真实探索需要尽可能大的公开中文词汇宇宙。

优先导入顺序：

1. `jieba` 内置公开中文词典：安装简单，适合快速获得数十万级中文词条；
2. THUOCL 中文词库：领域词汇，如财经、医学、地名、成语、IT 等；
3. 中文 Wikipedia titles：百科实体标题；
4. Wikidata 中文 labels / aliases：实体、多语言别名、类型 metadata；
5. OpenHowNet：义原、概念、中文词汇关系；
6. 用户自定义词表：笔记、论文关键词、项目术语、领域术语。

统一清洗规则：

- 每行一个 term；
- 去重并保留首次出现顺序；
- 保留 source metadata；
- 过滤空行、注释行、超长条目；
- 默认支持 `word frequency tag` 形式，只取第一列。

关键原则：

```text
embedding 模型只做 text -> vector；
vector -> word 必须通过大型候选词汇集合的最近邻检索完成。
```

## 4. 建议目录结构

```text
Project Hilbert/
  main.py
  README.md
  fiat_lux.md
  build_plan.md
  requirements.txt
  project_hilbert/
	__init__.py
	embeddings.py
	index.py
	chroma_store.py
	corpus.py
	clustering.py
	salience.py
	vector_logic.py
	schemas.py
	renderers.py
	cli.py
  data/
	concepts.zh.txt
	concepts.en.txt
  corpus/
	jieba_terms.txt
	zh_terms.txt
  .chroma/
	# Chroma persistent database
  tests/
	test_vector_logic.py
	test_salience.py
	test_clustering.py
```

## 5. 数据结构

### 5.1 ConceptRecord

```text
class ConceptRecord(BaseModel):
	id: str
	text: str
	metadata: dict[str, str | int | float] = {}
```

### 5.2 SearchResult

```text
class SearchResult(BaseModel):
	concept: str
	similarity: float
	distance: float
	cluster_id: int | None
	salient: float
	metadata: dict = {}
```

### 5.3 QueryResponse

```text
class QueryResponse(BaseModel):
	query: str
	mode: str
	results: list[SearchResult]
```

## 6. 核心模块设计

## 6.0 数学空间约定

Project Hilbert 的 embedding 空间必须显式定义内积、范数、距离和投影。默认约定如下。

设：

```text
H = R^d
d = embedding dimension，例如 BGE-M3 为 1024
v_a = E(a)
v_b = E(b)
```

默认内积：

```text
inner(a, b) = ⟨v_a, v_b⟩ = v_aᵀv_b
```

默认范数：

```text
norm(a) = ||v_a|| = sqrt(v_aᵀv_a)
```

默认 embedding 输出应归一化：

```text
||v_a|| = 1
inner(a, b) = cosine_similarity(a, b)
```

工程解释：

```text
inner =  1  -> 同向，最高 signed similarity
inner =  0  -> 近似正交，在当前度量下线性语义相关弱
inner = -1  -> 反向，最低 signed similarity，不是“高相似”
```

必须区分：

```text
signed_similarity(a, b) = inner(a, b)
axis_relatedness(a, b)  = abs(inner(a, b))
```

`-1` 只表示两个单位向量沿同一语义轴完全反向；它在 `abs(inner)` 下很高，但在 cosine similarity / signed similarity 下最低。默认搜索排序必须使用 `signed_similarity`，不能使用绝对值。

默认距离：

```text
distance(a, b) = 1 - inner(a, b)
```

归一化向量下：

```text
same direction -> 0
orthogonal     -> 1
opposite       -> 2
```

可选角距离：

```text
angular_distance(a, b) = arccos(clamp(inner(a, b), -1, 1))
```

投影：

```text
u_b = v_b / ||v_b||
proj_b(a) = inner(v_a, u_b) * u_b
```

语义剥离：

```text
a_without_b = v_a - proj_b(a)
```

用于未来实现：

- `A NOT B`；
- 去除背景语义；
- 多义词方向分解；
- 子空间投影。

加权内积：

```text
inner_M(x, y) = xᵀ M y
```

要求：

- `M` 应为正定或半正定矩阵；
- 初版不训练 `M`，只保留接口设计；
- 后续可用任务、领域或用户反馈学习 `M`；
- 如果使用 `M`，距离、投影、搜索排序必须明确记录 metric name。

注意：反义词不一定内积为负。现代 embedding 常把反义词放得很近，因为它们共享上下文和语义类型。例如“好”和“坏”都属于评价词，它们可能 cosine similarity 为正。不要把 `inner < 0` 简化理解为“反义”。

## 6.1 embeddings.py

职责：文本到向量。

接口：

```text
class EmbeddingModel:
	def encode(self, texts: list[str]) -> np.ndarray:
		...
```

默认模型建议：

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

原因：

- 支持中文与英文；
- 模型较小，适合本地 MVP；
- 可替换为 BGE / E5。

## 6.2 index.py

职责：保存概念向量并执行近邻搜索。

最小实现：

```text
class VectorIndex:
	def build(self, concepts: list[str]) -> None:
		...

	def search(self, query_vector: np.ndarray, top_n: int = 30) -> list[RawNeighbor]:
		...
```

相似度：优先使用 cosine similarity。

注意：向量最好归一化，方便使用点积近似 cosine similarity。

## 6.2.1 chroma_store.py

职责：使用 Chroma 保存大型词汇宇宙的向量、文本与 metadata，并提供近似最近邻搜索。

核心接口：

```text
class ChromaVectorStore:
	def build_from_file(self, concepts_path, embedder, batch_size, source, reset, limit) -> int:
		...

	def search_vector(self, query_vector, top_n) -> list[RawNeighbor]:
		...
```

Chroma collection 约定：

```text
persist_dir = .chroma
collection = project_hilbert_zh
metric = cosine
document = term text
metadata = {text, source, model, rank_in_source}
id = sha1(term)
```

查询流程：

```text
query text / vector logic expression
  -> BGE-M3 encode once
  -> Chroma collection.query(query_embeddings=[vector], n_results=search_k)
  -> local DBSCAN + salience on returned neighborhood
  -> display top_n
```

`search_k` 应大于 `top_n`，例如：

```text
search_k = 200
top_n = 30
```

这样 clustering 与 salient value 只在局部邻域上计算，不扫描全库。

## 6.2.2 corpus.py

职责：导入并清洗公开中文词表。

命令：

```powershell
python main.py import-jieba --out corpus/jieba_terms.txt
python main.py import-words corpus/raw/thuocl.txt corpus/raw/wiki_titles.txt --out corpus/zh_terms.txt
```

输出格式：

```text
苹果
苹果公司
北京
伦敦
人工智能
```

## 6.3 clustering.py

职责：对近邻结果自动分簇。

算法：DBSCAN。

输入：近邻向量。

输出：cluster labels。

伪代码：

```text
def cluster_neighbors(vectors: np.ndarray) -> list[int | None]:
	if len(vectors) < min_samples:
		return [None] * len(vectors)

	labels = DBSCAN(
		eps=eps,
		min_samples=min_samples,
		metric="cosine",
	).fit_predict(vectors)

	# DBSCAN 的 -1 表示 noise，可映射为 None
	return [None if label == -1 else int(label) for label in labels]
```

参数初值：

```text
eps = 0.25 到 0.40 之间调试
min_samples = 2 或 3
```

注意：

- 如果所有点都是 noise，则不分簇；
- 如果只有一个簇，也可以展示为一个 cluster；
- 后续可以引入 HDBSCAN，减少手动调 eps。

## 6.4 salience.py

职责：计算 salient value。

初版定义：基于局部密度的反向评分。

建议公式：

```text
candidate_density = mean(similarity(candidate, candidate_k_nearest_neighbors))
raw_salient = 1 - normalized(candidate_density)
salient = clamp(raw_salient, 0, 1)
```

可选增强：

```text
salient = alpha * sparsity + beta * query_similarity + gamma * boundary_score
```

其中：

- `sparsity`：候选点周围越稀疏越高；
- `query_similarity`：与查询越相关越高；
- `boundary_score`：位于多个簇边界时越高。

初始参数：

```text
alpha = 0.65
beta = 0.25
gamma = 0.10
```

边界情况：

- 概念库很小时，salient 可退化为 `1 - normalized(local_density)`；
- 如果无法计算密度，返回 `0.0` 或 `None`，但 MVP 建议返回 `0.0` 保持 JSON 稳定。

## 6.5 vector_logic.py

职责：解析并执行向量表达式。

支持表达式：

```text
北京 - 中国 + 英国
king - man + woman
苹果 - 水果 + 公司
```

解析规则：

- token 可以是中文、英文、短语；
- 操作符为 `+` 和 `-`；
- 从左到右执行；
- 初版不支持括号；
- 后续增加括号和乘法。

伪代码：

```text
def evaluate_expression(expression: str, embedder: EmbeddingModel) -> np.ndarray:
	tokens = parse_expression(expression)
	result = None
	current_op = "+"

	for token in tokens:
		if token in {"+", "-"}:
			current_op = token
			continue

		vector = embedder.encode([token])[0]
		if result is None:
			result = vector
		elif current_op == "+":
			result = result + vector
		elif current_op == "-":
			result = result - vector

	return normalize(result)
```

未来乘法研究方向：

1. element-wise multiplication：语义门控；
2. tensor product：组合两个概念形成更高阶关系；
3. learned relation matrix：把关系表示为矩阵变换；
4. composition operator：学习 `relation_a * relation_b -> relation_c`。

## 6.6 renderers.py

职责：输出 JSON / Markdown / 终端表格。

Markdown 输出模板：

```markdown
## Query: {query}

### Cluster {cluster_id}
| Concept | Similarity | Distance | Salient |
|---|---:|---:|---:|
| 苹果 | 0.91 | 0.09 | 0.33 |
```

JSON 输出必须稳定，便于测试。

## 6.7 cli.py

建议命令：

```powershell
python main.py build --concepts data/concepts.zh.txt
python main.py search "苹果" --top-n 30 --format markdown
python main.py logic "北京 - 中国 + 英国" --top-n 10 --format json
python main.py import-jieba --out corpus/jieba_terms.txt
python main.py build-chroma --model bge-m3 --concepts corpus/jieba_terms.txt --collection project_hilbert_zh --reset
python main.py search "苹果" --model bge-m3 --collection project_hilbert_zh --top-n 30 --search-k 200
python main.py logic "北京 - 中国 + 英国" --model bge-m3 --collection project_hilbert_zh --top-n 30 --search-k 200
```

CLI 参数：

```text
--model        embedding 模型名
--top-n        返回候选数量
--format       table/json/markdown
--cluster      是否启用聚类
--eps          DBSCAN eps
--min-samples  DBSCAN min_samples
--collection   Chroma collection 名称；提供后不再使用 --concepts 小列表
--persist-dir  Chroma 持久化目录
--search-k     内部召回候选数，用于聚类和 salience
```

## 7. MVP 执行顺序

### Step 1: 项目骨架

- 创建 package 目录；
- 创建 `requirements.txt`；
- 创建基础 CLI；
- 保证 `python main.py --help` 可运行。

### Step 2: Embedding 接入

- 实现 `EmbeddingModel`；
- 默认使用 multilingual MiniLM；
- 编写 smoke test：输入 `苹果` 能返回固定维度向量。

### Step 3: 概念库与索引

- 保留 `data/concepts.zh.txt` 作为 demo；
- 新增 `corpus.py` 导入公开中文词表；
- 新增 `chroma_store.py` 构建 Chroma collection；
- 支持 `import-jieba`、`import-words`、`build-chroma`。

### Step 4: Search

- 实现 query -> vector -> nearest concepts；
- 输出 similarity 和 distance；
- 添加基础测试。

### Step 5: Clustering

- 对 top-n 结果执行 DBSCAN；
- 输出 cluster_id；
- 对 DBSCAN 全 noise 情况做降级处理。

### Step 6: Salience

- 实现局部密度；
- 归一化到 `[0, 1]`；
- 加入 SearchResult。

### Step 7: Vector Logic

- 实现 `+` / `-` 表达式解析；
- 对结果向量执行 search；
- 添加经典测试用例。

### Step 8: 文档与示例

- 创建 `README.md`；
- 添加 quickstart；
- 添加 `苹果` 与 `北京 - 中国 + 英国` 示例。

## 8. 测试计划

### 单元测试

- `parse_expression`：
  - `A + B`
  - `A - B + C`
  - 多空格；
  - 非法表达式；
- `normalize`：零向量处理；
- `salience`：密集点低分、稀疏点高分；
- `clustering`：
  - 少量点；
  - 单簇；
  - 多簇；
  - 全 noise。

### 集成测试

- 构建小型概念库：

```text
苹果
香蕉
梨
橙子
谷歌
微软
英伟达
北京
中国
伦敦
英国
```

- 查询 `苹果`，应返回水果与科技公司相关项；
- 表达式 `北京 - 中国 + 英国`，应优先接近 `伦敦` 或相关英国城市概念。

### 性能测试

- 100 个概念；
- 1,000 个概念；
- 10,000 个概念；
- 100,000 个概念；
- Chroma collection count、写入速度、查询延迟；
- 记录 embedding 时间、索引时间、查询时间。

## 9. 工程约束

- 所有公共函数必须有类型标注；
- JSON 输出字段名保持稳定；
- 不在核心模块中打印，应由 CLI / renderer 负责展示；
- embedding 模型必须可替换；
- 小规模 MVP 先以内存实现，不提前复杂化；
- 任何随机过程需要设置 random seed；
- 文档示例应可复制运行。
- `data/concepts.zh.txt` 不得作为真实探索的默认上限；大型探索必须走 Chroma collection。

## 10. 风险与注意事项

### 10.1 Embedding 偏差

不同模型的向量空间结构不同，类比结果可能差异很大。

缓解：

- 支持模型切换；
- 在结果中记录 model name；
- 建立 benchmark examples。

### 10.2 多义词问题

如 `苹果` 同时代表水果和公司。

缓解：

- 使用 clustering 展示多语义簇；
- 支持上下文查询，如 `苹果 水果`、`苹果 公司`。

### 10.3 DBSCAN 参数敏感

`eps` 不合适会导致全 noise 或过度合并。

缓解：

- 暴露 CLI 参数；
- 自动尝试多个 eps；
- 后续引入 HDBSCAN。

### 10.4 Salience 解释性

salient value 不是客观真理，只是探索指标。

缓解：

- 文档中明确公式；
- 输出 density 等辅助字段可选开启；
- 保持指标可替换。

## 11. 推荐初始概念库

`data/concepts.zh.txt` 可先包含：

```text
苹果
香蕉
梨
橙子
葡萄
西瓜
谷歌
微软
英伟达
亚马逊
特斯拉
芯片
人工智能
智能手机
北京
中国
伦敦
英国
巴黎
法国
东京
日本
国王
女王
男人
女人
医生
医院
教师
学校
```

## 12. Definition of Done

MVP 完成标准：

- `python main.py --help` 可运行；
- 可以从文本文件加载概念库；
- 可以构建或缓存向量；
- `search "苹果"` 返回带 similarity、cluster、salient 的结果；
- `logic "北京 - 中国 + 英国"` 返回近邻结果；
- JSON 输出可被测试稳定解析；
- Markdown 输出适合放入 GitHub issue / README；
- 至少包含 10 个单元测试；
- README 包含安装、运行、示例。

## 13. 给智能体的执行提示

优先级：

1. 先做可运行的最小闭环；
2. 不要过早引入大型框架；
3. 每完成一个模块就写测试；
4. 所有示例命令必须在 Windows PowerShell 下可运行；
5. 保留清晰扩展点，不把模型、索引、聚类、渲染耦合在一起；
6. 如果真实 embedding 下载过慢，可先实现 deterministic mock embedder 用于测试；
7. 测试不能依赖网络。

## 14. 下一步建议

建议下一次工程实现从以下文件开始：

```text
requirements.txt
project_hilbert/schemas.py
project_hilbert/vector_logic.py
project_hilbert/salience.py
tests/test_vector_logic.py
tests/test_salience.py
```

先让核心纯函数测试通过，再接入真实 embedding 模型。

