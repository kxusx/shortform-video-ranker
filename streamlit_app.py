"""Streamlit UI for FAISS-backed short-form video search."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import google.generativeai as genai
import streamlit as st
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from transformers import CLIPModel, CLIPProcessor


PROJECT_ROOT = Path(__file__).resolve().parent
INDEX_ROOT = PROJECT_ROOT / "notebook_artifacts" / "index"

FAISS_INDEX_PATH = PROJECT_ROOT / "faiss" / "faiss.index"
FAISS_META_PATH = PROJECT_ROOT / "faiss" / "faiss_scene_meta.json"
FAISS_SCENE_IDS_PATH = PROJECT_ROOT / "faiss" / "faiss_scene_ids.json"
VIDEO_METADATA_PATH = INDEX_ROOT / "video_metadata.json"
SCENE_RECORDS_PATH = INDEX_ROOT / "scene_records.json"
MSRVTT_CAPTIONS_PATH = PROJECT_ROOT / "msrvtt_train_7k.json"

load_dotenv()


def _read_json(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text()
    return json.loads(text) if text else {}


@st.cache_resource(show_spinner=False)
def load_clip() -> Tuple[CLIPProcessor, CLIPModel, str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    model.eval()
    return processor, model, device


@st.cache_resource(show_spinner=False)
def load_faiss() -> Tuple[faiss.Index, List[Dict], List[str]]:
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    meta = json.loads(FAISS_META_PATH.read_text())
    scene_ids = json.loads(FAISS_SCENE_IDS_PATH.read_text())
    return index, meta, scene_ids


@st.cache_data(show_spinner=False)
def load_metadata() -> Dict:
    return _read_json(VIDEO_METADATA_PATH)


@st.cache_data(show_spinner=False)
def load_captions() -> Dict[str, List[str]]:
    if not MSRVTT_CAPTIONS_PATH.exists():
        return {}
    records = json.loads(MSRVTT_CAPTIONS_PATH.read_text())
    caption_index: Dict[str, List[str]] = {}
    for item in records:
        captions = item.get("caption") or []
        if isinstance(captions, str):
            captions = [captions]
        video_id = str(item.get("video_id", "")).strip()
        video_name = item.get("video")
        keys = []
        if video_id:
            keys.append(video_id)
        if video_name:
            name = Path(video_name).name
            keys.extend([name, Path(name).stem])
        for key in keys:
            if not key:
                continue
            store = caption_index.setdefault(key, [])
            for cap in captions:
                if cap not in store:
                    store.append(cap)
    return caption_index


def _strip_code_fences(text: str) -> str:
    trimmed = text.strip()
    if trimmed.startswith("```") and trimmed.endswith("```"):
        trimmed = trimmed[3:-3].strip()
        if trimmed.lower().startswith("json"):
            trimmed = trimmed[4:].strip()
    return trimmed


def rerank_with_gemini(
    search_output: Dict,
    query: str,
    *,
    top_results: int,
    api_key: str | None,
    model_name: str = "gemini-2.5-pro",
) -> Dict:
    videos = search_output.get("video_results", [])
    if not videos:
        return {"reranked": [], "raw_response": "", "prompt": ""}
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError("Gemini API key missing. Set GEMINI_API_KEY or enter it in the sidebar.")

    genai.configure(api_key=key)
    model = genai.GenerativeModel(model_name)

    payload = []
    for idx, video in enumerate(videos, 1):
        payload.append(
            {
                "rank": idx,
                "video_id": video.get("video_id"),
                "video_uri": video.get("video_uri"),
                "score": video.get("score"),
                "scene_frames": video.get("scene_frames"),
                "captions": video.get("captions", []),
            }
        )

    instructions = f'Rank the best {top_results} clips for the query "{query}".'
    prompt = (
        instructions
        + "\nReturn JSON under key reranked (best to worst). "
        + "Each item must include video_id, video_uri, and a short reason."
        + "\nCandidates:\n"
        + json.dumps(payload, indent=2)
    )

    response = model.generate_content(prompt)
    text = getattr(response, "text", "") or ""
    if not text and hasattr(response, "candidates"):
        text = "".join(getattr(part, "text", "") for part in response.candidates)

    cleaned = _strip_code_fences(text)
    reranked = []
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            reranked = parsed.get("reranked", [])
        elif isinstance(parsed, list):
            reranked = parsed
    except json.JSONDecodeError as exc:
        raise ValueError(f"Gemini response was not valid JSON: {exc}") from exc

    reranked = reranked or []
    return {"reranked": reranked[:top_results], "raw_response": text, "prompt": prompt}


def merge_reranked(videos: List[Dict], reranked: List[Dict]) -> List[Dict]:
    """Return videos ordered by reranked list while keeping metadata."""

    def _match(candidate: Dict) -> Dict | None:
        vid = candidate.get("video_id")
        uri = candidate.get("video_uri")
        for video in videos:
            if vid is not None and str(video.get("video_id")) == str(vid):
                return video
        if uri:
            for video in videos:
                if video.get("video_uri") == uri:
                    return video
        return None

    seen_keys = set()
    ordered: List[Dict] = []
    for item in reranked:
        match = _match(item)
        if not match:
            continue
        key = f"{match.get('video_id')}|{match.get('video_uri')}"
        if key in seen_keys:
            continue
        merged = dict(match)
        reason = item.get("reason")
        if reason:
            merged["reason"] = reason
        ordered.append(merged)
        seen_keys.add(key)

    for video in videos:
        key = f"{video.get('video_id')}|{video.get('video_uri')}"
        if key not in seen_keys:
            ordered.append(video)
    return ordered


def search_with_faiss(text: str, top_scene_hits: int = 30, top_videos: int = 10) -> Dict:
    processor, model, device = load_clip()
    index, scene_meta, _ = load_faiss()
    video_metadata = load_metadata()

    text_inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
    with torch.inference_mode():
        text_features = model.get_text_features(**text_inputs)
    text_features = F.normalize(text_features, dim=-1).cpu().numpy().astype("float32")
    k = min(max(top_scene_hits, 1), index.ntotal)
    scores, indices = index.search(text_features, k)

    scene_hits: List[Dict] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        meta = scene_meta[idx]
        scene_hits.append(
            {
                "scene_id": meta["scene_id"],
                "video_id": meta["video_id"],
                "score": float(score),
                "start_frame": meta.get("start_frame"),
                "end_frame": meta.get("end_frame"),
            }
        )

    video_scores: Dict[str, Dict] = {}
    for hit in scene_hits:
        vid = hit["video_id"]
        current = video_scores.get(vid)
        if current is None or hit["score"] > current["score"]:
            video_scores[vid] = hit

    ranked = sorted(video_scores.values(), key=lambda item: item["score"], reverse=True)[
        :top_videos
    ]
    results = []
    for hit in ranked:
        md = video_metadata.get(hit["video_id"], {})
        results.append(
            {
                "video_id": hit["video_id"],
                "video_uri": md.get("video_uri"),
                "video_duration": md.get("video_duration"),
                "score": hit["score"],
                "scene_id": hit["scene_id"],
                "scene_frames": (hit["start_frame"], hit["end_frame"]),
            }
        )
    return {"scene_hits": scene_hits, "video_results": results}


def attach_captions(search_output: Dict) -> Dict:
    captions = load_captions()
    enriched_videos = []
    for hit in search_output.get("video_results", []):
        keys = []
        vid = hit.get("video_id")
        if vid:
            keys.append(str(vid))
        video_uri = hit.get("video_uri")
        if video_uri:
            name = Path(video_uri).name
            keys.extend([name, Path(name).stem])
        caption_candidates: List[str] = []
        for key in keys:
            matched = captions.get(key)
            if matched:
                caption_candidates = matched
                break
        enriched = dict(hit)
        enriched["captions"] = caption_candidates
        enriched_videos.append(enriched)
    return {"scene_hits": search_output.get("scene_hits", []), "video_results": enriched_videos}


def _render_video_card(result: Dict):
    video_uri = result.get("video_uri")
    caption_list = result.get("captions") or []
    reason = result.get("reason") or (caption_list[0] if caption_list else f"Score: {result.get('score', 0):.4f}")
    label = f"Video {result.get('video_id', 'N/A')} • {result.get('score', 0):.3f}"

    card = st.container()
    with card:
        st.markdown(f"**{label}**")
        if video_uri and Path(video_uri).exists():
            st.video(str(video_uri))
        else:
            st.write("Video file not found locally.")
        st.caption(reason)
        frames = result.get("scene_frames")
        if frames:
            st.caption(f"Scene frames: {frames[0]} – {frames[1]}")


def main():
    st.set_page_config(page_title="Shortform Video Search", page_icon="🎬", layout="wide")
    st.title("Shortform Video Search")
    st.write("Type a description and press enter to fetch the best matching clips.")

    with st.sidebar:
        st.header("Search Options")
        top_videos = st.slider("Videos to show", min_value=1, max_value=20, value=6)
        top_scenes = st.slider("Scene hits to scan", min_value=5, max_value=100, value=40, step=5)
        st.divider()
        st.header("Gemini reranker")
        enable_rerank = st.toggle("Use Gemini reranker", value=True)
        rerank_limit = st.slider(
            "Top reranked videos",
            min_value=1,
            max_value=top_videos,
            value=min(3, top_videos),
            disabled=not enable_rerank,
        )
        gemini_model_name = st.selectbox(
            "Model",
            ["gemini-2.5-pro"],
            index=0,
            disabled=not enable_rerank,
        )
        # gemini_key_input = st.text_input(
        #     "Gemini API Key",
        #     placeholder="Leave blank to read GEMINI_API_KEY",
        #     type="password",
        #     disabled=not enable_rerank,
        # )
        # if enable_rerank:
        #     st.caption("Provide an API key or export GEMINI_API_KEY before launching Streamlit.")

        try:
            _, _, device = load_clip()
            index, _, _ = load_faiss()
            st.success(
                f"Ready • CLIP on {device.upper()} • {index.ntotal} scenes indexed",
                icon="✅",
            )
        except Exception as exc:  # pragma: no cover - surfaced in UI
            st.error(f"Failed to load search artifacts: {exc}")
            st.stop()

    query = st.text_input(
        "Describe the video you're looking for",
        placeholder="Example: a person talking to the camera about cooking",
    ).strip()

    if not query:
        st.info("Enter a query to run the search.")
        return

    rerank_used = False

    with st.spinner("Searching..."):
        raw_results = search_with_faiss(query, top_scene_hits=top_scenes, top_videos=top_videos)
        enriched = attach_captions(raw_results)
        videos = enriched.get("video_results", [])
        if enable_rerank:
            effective_key = (os.getenv("GEMINI_API_KEY", "")).strip()
            if not effective_key:
                st.warning("Gemini reranker enabled but no API key supplied; showing FAISS order.")
            else:
                try:
                    reranked_payload = rerank_with_gemini(
                        enriched,
                        query,
                        top_results=rerank_limit,
                        api_key=effective_key,
                        model_name=gemini_model_name,
                    )
                    reranked_items = reranked_payload.get("reranked", [])
                    if reranked_items:
                        videos = merge_reranked(videos, reranked_items)
                        rerank_used = True
                    else:
                        st.info("Gemini reranker returned no items; showing FAISS order.")
                except Exception as exc:
                    st.error(f"Gemini rerank failed: {exc}")

    if not videos:
        st.warning("No matches found. Try broadening the description.")
        return

    st.subheader(f"Results for “{query}”")
    if rerank_used:
        st.caption("Ordered by Gemini reranker.")
    for item in videos:
        _render_video_card(item)


if __name__ == "__main__":
    main()
