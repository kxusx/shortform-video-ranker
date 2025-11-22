# Shortform Video Search Playground
This project is a POC for short-form video retrieval and reranking. Everything lives in `code.ipynb`, making it easy to iterate on indexing, searching, and experimenting with LLM rerankers.

## How the pipeline works
1. **Dataset ingestion**  
   Point the notebook at your local dataset (currently `data/1/TrainValVideo`, which contains MSRVTT clips). The helper `list_video_files` function discovers videos recursively and feeds them to the indexer.

2. **Scene detection**  
   Each video is split into scenes with [PySceneDetect](https://pyscenedetect.readthedocs.io/) by monitoring frame-to-frame visual changes. If PySceneDetect cannot find cuts, the whole video is treated as one continuous scene so we never skip content.

3. **Frame sampling & CLIP embeddings**  
   For every detected scene, we sample a handful of evenly spaced frames. Each sampled frame goes through the Hugging Face `CLIPProcessor` + `CLIPModel` to produce image embeddings. We average the tensors to get a single per-scene embedding and store it on disk (`notebook_artifacts/index/scene_tensors/<uuid>.pt`). This mirrors the original project’s caching strategy and keeps re-computation cheap.

4. **Notebook-friendly indexes**  
   The notebook serializes metadata into JSON:
   - `video_metadata.json`: video URI and duration per video ID
   - `video_scenes.json`: mapping of video IDs to their scene IDs
   - `scene_records.json`: tensor paths and frame spans per scene ID
   - `uri_to_id.json`: guards against double indexing the same file

5. **Searching with CLIP**  
   The initial search path iterates over indexed videos, reloads their cached tensors, and compares them against a CLIP text embedding to produce a similarity score. This is useful for debugging and for very small datasets.

## FAISS acceleration
As soon as you have more than a handful of clips, iterating over every scene at query time becomes slow. To fix that, the notebook introduces a FAISS-based scene index:

- **Scene embedding matrix**: After indexing videos, the FAISS section re-embeds each stored scene tensor through CLIP’s image tower (to get 512-dim image features) and stacks them into an `N x D` matrix.
- **IndexFlatIP + metadata**: We build a FAISS `IndexFlatIP` over that matrix and keep parallel lists of scene IDs and metadata (video ID, frame bounds). Everything lives in memory for fast retrieval, but helper cells also let you persist to disk (`faiss.index`, `faiss_scene_meta.json`, `faiss_scene_ids.json`) and reload later.
- **Fast queries**: A text prompt is embedded once via CLIP’s text encoder, then we call `faiss_index.search` for the top-k scene matches. Those hits are aggregated back to videos, giving you an ordered list of clips in milliseconds—even on CPU-only machines.

## Reranking with LLMs
The notebook finishes with a stub showing how to pass the FAISS/CLIP results into an LLM for a second-pass rerank. You can plug in OpenAI, Azure, or any local model, enriching prompts with extra metadata (ASR transcripts, scene descriptions, etc.) to get more nuanced results.

## Getting started
1. Create a virtual environment and install the dependencies listed in the notebook (PyTorch, transformers, opencv-python, scenedetect, faiss-cpu, tqdm, pillow, etc.).  
2. Open `code.ipynb` and run cells sequentially:
   - Configure paths and inspect the dataset
   - Index a subset of videos to verify everything works
   - Build the FAISS index and optionally persist it
   - Run searches and, if desired, hook up your LLM reranker
3. Scale up ingestion by expanding the subset or using `bulk_index` with a larger limit.

This setup gives you a reproducible sandbox for exploring CLIP-based video retrieval, fast ANN search via FAISS, and downstream reranking—all within a single notebook.
