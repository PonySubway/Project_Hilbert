# Corpus directory

This directory is for generated or user-provided lexicons used to build Chroma collections.

Generated files such as `jieba_terms.txt` are intentionally ignored by Git because they can be large and can be recreated:

```powershell
python main.py import-jieba --out corpus\jieba_terms.txt
```

Then build a local Chroma collection:

```powershell
python main.py build-chroma --model bge-m3 --concepts corpus\jieba_terms.txt --collection project_hilbert_zh --persist-dir .\.chroma --batch-size 64 --reset
```

