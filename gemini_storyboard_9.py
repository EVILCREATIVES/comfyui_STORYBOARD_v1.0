import json
import base64
import io
from typing import Tuple, List, Dict, Any

import requests
import numpy as np
from PIL import Image

try:
    import torch
except Exception:
    torch = None


GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


def _pil_to_comfy_image(pil: Image.Image) -> np.ndarray:
    """
    ComfyUI IMAGE type is numpy float32 in [0..1], shape [H, W, 3]
    """
    pil = pil.convert("RGB")
    arr = np.asarray(pil).astype(np.float32) / 255.0
    return arr


def _stack_batch(images: List[np.ndarray]) -> np.ndarray:
    """
    Stack [H,W,3] images into [B,H,W,3]
    Assumes all same size (we enforce by resizing if needed).
    """
    return np.stack(images, axis=0)


def _tile_grid_3x3(batch: np.ndarray) -> np.ndarray:
    """
    batch: [9, H, W, 3] -> grid: [3H, 3W, 3]
    """
    assert batch.shape[0] == 9, "Expected exactly 9 images in batch"
    H, W = batch.shape[1], batch.shape[2]
    grid = np.zeros((H * 3, W * 3, 3), dtype=batch.dtype)

    idx = 0
    for r in range(3):
        for c in range(3):
            grid[r * H:(r + 1) * H, c * W:(c + 1) * W, :] = batch[idx]
            idx += 1
    return grid


def _encode_ref_image_png_b64(image_np: np.ndarray) -> str:
    """
    image_np: [H, W, 3] float32 [0..1]
    returns raw base64 PNG data (no data: prefix)
    """
    pil = Image.fromarray(np.clip(image_np * 255.0, 0, 255).astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _gemini_generate_content(
    api_key: str,
    model: str,
    body: Dict[str, Any],
    timeout_s: int = 120,
) -> Dict[str, Any]:
    url = f"{GEMINI_BASE}/{model}:generateContent"
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
    }
    r = requests.post(url, headers=headers, data=json.dumps(body), timeout=timeout_s)
    if r.status_code >= 400:
        raise RuntimeError(f"Gemini API error {r.status_code}: {r.text}")
    return r.json()


def _extract_text_from_generate_content(resp: Dict[str, Any]) -> str:
    # Typical: candidates[0].content.parts[0].text  [oai_citation:1‡Google AI for Developers](https://ai.google.dev/api/generate-content)
    try:
        parts = resp["candidates"][0]["content"]["parts"]
    except Exception:
        raise RuntimeError(f"Unexpected Gemini response (no candidates/content/parts): {json.dumps(resp)[:2000]}")
    texts = []
    for p in parts:
        if "text" in p and p["text"]:
            texts.append(p["text"])
    return "\n".join(texts).strip()


def _extract_first_image_b64(resp: Dict[str, Any]) -> Tuple[str, str]:
    """
    Returns (mime_type, base64_data) for the first inline image part.
    Gemini native image gen returns inlineData/inline_data in parts.  [oai_citation:2‡Google AI for Developers](https://ai.google.dev/gemini-api/docs/image-generation)
    """
    try:
        parts = resp["candidates"][0]["content"]["parts"]
    except Exception:
        raise RuntimeError(f"Unexpected Gemini response (no candidates/content/parts): {json.dumps(resp)[:2000]}")

    # Some responses use inlineData (camelCase) in REST; others inline_data (snake_case).
    for p in parts:
        if "inlineData" in p and p["inlineData"]:
            mime = p["inlineData"].get("mimeType") or p["inlineData"].get("mime_type") or "image/png"
            data = p["inlineData"].get("data")
            if data:
                return mime, data
        if "inline_data" in p and p["inline_data"]:
            mime = p["inline_data"].get("mime_type") or p["inline_data"].get("mimeType") or "image/png"
            data = p["inline_data"].get("data")
            if data:
                return mime, data

    raise RuntimeError(
        "No inline image data found in response parts. "
        "Make sure you're using an image-capable model (e.g. gemini-2.5-flash-image or gemini-3-pro-image-preview) "
        "and responseModalities includes Image if needed."
    )


def _decode_image_b64_to_pil(b64_data: str) -> Image.Image:
    raw = base64.b64decode(b64_data)
    return Image.open(io.BytesIO(raw))


class GeminiStoryboard9:
    """
    1) Gemini text model -> 9 beats JSON
    2) Gemini image model -> 9 independent images (1 per beat)
    3) Output batch + grid + beats JSON
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gemini_api_key": ("STRING", {"multiline": False}),
                "story_text": ("STRING", {"multiline": True}),
                "beats_model": ("STRING", {"default": "gemini-2.5-flash"}),
                "image_model": ("STRING", {"default": "gemini-2.5-flash-image"}),
                "aspect_ratio": ("STRING", {"default": "16:9"}),  # per docs  [oai_citation:3‡Google AI for Developers](https://ai.google.dev/gemini-api/docs/image-generation)
                "image_size": ("STRING", {"default": ""}),        # for gemini-3-pro-image-preview: "1K"|"2K"|"4K"  [oai_citation:4‡Google AI for Developers](https://ai.google.dev/gemini-api/docs/image-generation)
                "strict_identity": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            },
            "optional": {
                "reference_image": ("IMAGE",),  # ComfyUI IMAGE batch; we use first frame
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("images_batch_9", "grid_3x3", "beats_json")
    FUNCTION = "run"
    CATEGORY = "Gemini"

    def run(
        self,
        gemini_api_key: str,
        story_text: str,
        beats_model: str,
        image_model: str,
        aspect_ratio: str,
        image_size: str,
        strict_identity: bool,
        seed: int,
        reference_image=None,
    ):
        # ---- Reference image handling (optional)
        ref_b64 = None
        if reference_image is not None:
            # reference_image is [B,H,W,3] float32; use first
            ref0 = reference_image[0] if reference_image.ndim == 4 else reference_image
            ref_b64 = _encode_ref_image_png_b64(ref0)

        # ---- 1) Generate 9 beats as strict JSON
        identity_line = (
            "Continuity rule: preserve the SAME main character identity and consistent style across all beats."
            if strict_identity else
            "Continuity rule: preserve a consistent style across all beats (character may vary)."
        )

        beats_prompt = (
            "You are a storyboard planner.\n"
            "Generate EXACTLY 9 storyboard beats.\n"
            "Return ONLY valid JSON (no markdown, no commentary).\n"
            "Schema:\n"
            "{\"beats\":[{\"id\":1,\"shot\":\"\",\"camera\":\"\",\"location\":\"\",\"action\":\"\",\"mood\":\"\",\"image_prompt\":\"\"}]}\n"
            "Rules:\n"
            "- ids must be 1..9\n"
            "- each beat must be visually different (different shot/action)\n"
            f"- {identity_line}\n"
            "- image_prompt must be a self-contained image generation prompt\n"
            "\n"
            "STORY:\n"
            f"{story_text}\n"
        )

        beats_body = {
            "contents": [{
                "role": "user",
                "parts": [{"text": beats_prompt}]
            }],
            "generationConfig": {
                "response_mime_type": "application/json"
            }
        }
        # Optionally include reference image to inform beat planning
        if ref_b64:
            beats_body["contents"][0]["parts"].append({
                "inline_data": {"mime_type": "image/png", "data": ref_b64}
            })

        beats_resp = _gemini_generate_content(gemini_api_key, beats_model, beats_body, timeout_s=120)
        beats_json_text = _extract_text_from_generate_content(beats_resp)

        try:
            beats_obj = json.loads(beats_json_text)
            beats = beats_obj["beats"]
        except Exception as e:
            raise RuntimeError(
                "Failed to parse beats JSON. Raw model output:\n"
                f"{beats_json_text}\n\nError: {e}"
            )

        if not isinstance(beats, list) or len(beats) != 9:
            raise RuntimeError(f"Beats must be a list of exactly 9 items. Got: {type(beats)} len={len(beats) if isinstance(beats, list) else 'n/a'}")

        # ---- 2) For each beat, generate one independent image
        out_images: List[np.ndarray] = []

        for i, beat in enumerate(beats, start=1):
            image_prompt = beat.get("image_prompt", "")
            if not image_prompt.strip():
                # fallback: construct from fields
                image_prompt = f"{beat.get('shot','')}, {beat.get('camera','')}, {beat.get('location','')}, {beat.get('action','')}, {beat.get('mood','')}"

            final_prompt = (
                f"Frame {i}/9.\n"
                f"{identity_line}\n"
                "No text, no watermark, no logos.\n"
                f"{image_prompt}\n"
            )

            img_body = {
                "contents": [{
                    "role": "user",
                    "parts": [{"text": final_prompt}]
                }],
                "generationConfig": {
                    # Ask explicitly for Image modality (per REST example).  [oai_citation:5‡Google AI for Developers](https://ai.google.dev/gemini-api/docs/image-generation)
                    "responseModalities": ["Image"],
                    "imageConfig": {
                        "aspectRatio": aspect_ratio,
                    }
                }
            }

            # gemini-3-pro-image-preview supports imageSize like 1K/2K/4K  [oai_citation:6‡Google AI for Developers](https://ai.google.dev/gemini-api/docs/image-generation)
            if image_size.strip():
                img_body["generationConfig"]["imageConfig"]["imageSize"] = image_size.strip()

            # Provide reference image as conditioning (optional but recommended)
            if ref_b64:
                img_body["contents"][0]["parts"].append({
                    "inline_data": {"mime_type": "image/png", "data": ref_b64}
                })

            img_resp = _gemini_generate_content(gemini_api_key, image_model, img_body, timeout_s=240)
            mime, b64 = _extract_first_image_b64(img_resp)
            pil = _decode_image_b64_to_pil(b64)

            # Normalize size across outputs (Gemini should match aspect, but be safe)
            arr = _pil_to_comfy_image(pil)
            out_images.append(arr)

        # Ensure same size (resize to first)
        H0, W0 = out_images[0].shape[0], out_images[0].shape[1]
        normalized: List[np.ndarray] = []
        for arr in out_images:
            if arr.shape[0] != H0 or arr.shape[1] != W0:
                pil = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), mode="RGB")
                pil = pil.resize((W0, H0), resample=Image.BICUBIC)
                normalized.append(_pil_to_comfy_image(pil))
            else:
                normalized.append(arr)

        batch9 = _stack_batch(normalized)          # [9,H,W,3]
        grid = _tile_grid_3x3(batch9)              # [3H,3W,3]

        return (batch9, grid[np.newaxis, ...], beats_json_text)


NODE_CLASS_MAPPINGS = {
    "GeminiStoryboard9": GeminiStoryboard9,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiStoryboard9": "Gemini Storyboard (9 beats → 9 images + 3x3 grid)",
}
