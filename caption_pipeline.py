import json, os, random, pandas as pd
from typing import Dict, Any, List
import google.generativeai as genai
from google.api_core.exceptions import NotFound, InvalidArgument

# To see which metadata keys were actually considered in the final output
ALLOWED_FIELDS = {"bridge_type", "span_length", "deck_width", "bridge_height", "missing_components", "num_piers"}

# === Build LLM prompt ===
def build_prompt(meta: Dict[str, Any], k: int = 1) -> str:
    """
    Produces: caption + 3 single-round QAs + 1 multi-round conversation.
    """
    tone = random.choice([
        "write in a factual yet vivid tone",
        "describe the object precisely but avoid redundancy",
        "use clear, professional phrasing appropriate for academic datasets"
    ])

    meta_llm = {fld: meta.get(fld) for fld in [
        "domain", "bridge_type", "span_length", "deck_width",
        "bridge_height", "missing_components", "num_piers", "target_tokens"
    ]}
    meta_llm["target_tokens"] = meta_llm.get("target_tokens", 80)

    return (
        "You are a PointLLM-style captioning assistant. "
        "Analyze the given 3D bridge metadata and create a rich structured JSON response.\n\n"
        "DO NOT mention data quality issues (e.g., occlusion, sparsity, missing scan parts). "
        "Describe only the object itself — its geometry, structure and potential usage.\n\n"
        "Return a valid JSON with this schema:\n"
        "{\n"
        '  "caption": "A detailed paragraph (50–100 words) describing the 3D object",\n'
        '  "single_conversation": [\n'
        '    {"Q": "...", "A": "..."},  # 3 single-round Q&A pairs\n'
        '  ],\n'
        '  "multi_conversation": [\n'
        '    {"Q1": "...", "A1": "...", "Q2": "...", "A2": "...", "Q3": "...", "A3": "..."}\n'
        '  ]\n'
        "}\n\n"
        "Guidelines:\n"
        "- The caption should cover geometry, materials, dimensions, and function.\n"
        "- Avoid speculative information.\n"
        "- Use clear technical language.\n"
        "- The Q&A pairs should highlight meaningful features (e.g., structure, purpose).\n"
        "- The multi-round Q&A should show logical continuity.\n\n"
        f"{tone}.\n\n"
        f"METADATA_JSON:\n{json.dumps(meta_llm, ensure_ascii=False)}"
    )

# === Extract usable text ===
def _extract_text(resp):
    if not resp:
        return None
    try:
        if getattr(resp, "text", None):
            return resp.text
    except:
        pass
    try:
        parts = getattr(resp.candidates[0], "content", None)
        if parts and getattr(parts, "parts", None):
            buf = [getattr(p, "text", None) for p in parts.parts if getattr(p, "text", None)]
            if buf:
                return "".join(buf).strip()
    except:
        pass
    return None

# === Gemini API Call ===
def call_gemini(prompt: str) -> str:
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY environment variable is not set.")

    genai.configure(api_key=key, transport="rest")
    models = ["models/gemini-2.5-flash", "models/gemini-2.5-pro"]

    safety = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]

    last_err = None
    for model in models:
        try:
            m = genai.GenerativeModel(model)
            try:
                token_info = m.count_tokens(prompt)
                in_tokens = int(getattr(token_info, "total_tokens", 0) or 0)
                print(f"Input tokens for {model}: {in_tokens}")
            except Exception:
                in_tokens = 0

            max_out = min(2048, 8000 - in_tokens - 200) if in_tokens and in_tokens < 7000 else 1024
            gen = lambda mx: m.generate_content(
                prompt,
                generation_config=dict(
                    temperature=0.7,
                    top_p=0.9,
                    max_output_tokens=mx,
                    response_mime_type="application/json"
                ),
                safety_settings=safety
            )

            resp = gen(max_out)
            text = _extract_text(resp)

            if not (text and text.strip()):
                resp = gen(min(max_out * 2, 4096))
                text = _extract_text(resp)

            if text and text.strip():
                return text

            finish_reason = getattr(resp.candidates[0], "finish_reason", None) if getattr(resp, "candidates", None) else None
            print(f"⚠️ No usable text from {model}. finish_reason={finish_reason}")
            last_err = RuntimeError(f"No usable text from {model}.")
        except (NotFound, InvalidArgument) as e:
            last_err = e
            print(f"Model error for {model}: {e}")
        except Exception as e:
            last_err = e
            print(f"Unexpected error for {model}: {e}")

    raise RuntimeError(f"All Gemini models failed. Last error: {last_err}")

# === Metadata validation ===
def validate_against_metadata(caption: str, meta: Dict[str, Any]) -> List[str]:
    errs = []
    if meta.get("span_length") and "m" not in caption:
        errs.append("Length unit 'm' not found in caption.")
    if meta.get("num_piers") and "pier" not in caption.lower():
        errs.append("Piers not mentioned in caption.")
    return errs

# === Main pipeline ===
def generate_captions(meta: Dict[str, Any]) -> Dict[str, Any]:
    raw = call_gemini(build_prompt(meta))
    try:
        data = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
    except Exception as e:
        raise RuntimeError(f"JSON parse failed: {e}. Raw preview: {raw[:600]}")

    for k in ["caption", "single_conversation", "multi_conversation"]:
        if k not in data:
            raise RuntimeError(f"Missing '{k}' in output. Got keys: {list(data.keys())}")

    issues = validate_against_metadata(data["caption"], meta)
    if issues:
        print(f"⚠️ Metadata validation issues: {issues}")

    return {
        "id": meta["id"],
        "domain": meta.get("domain", "bridge"),
        "output": data,
        "metadata_used": sorted(ALLOWED_FIELDS & set(meta.keys())),
        "validation_issues": issues
    }

# === Entry point ===
if __name__ == "__main__":

    df = pd.read_csv("metadata.csv")

    # Convert one row into dict
    first = df.iloc[0].to_dict()
    meta = {
        "id": str(first.get("BridgeID")),
        "bridge_type": first.get("Bridge_Type"),
        "span_length": first.get("SpanLength"),
        "deck_width": first.get("DeckWidth"),
        "bridge_height": first.get("BridgeHeight"),
        "missing_components": first.get("MissingComponents"),
        "num_piers": first.get("NumberOfPiers"),
        "domain": "bridges"
    }

    out_path = "output.json"
    result = generate_captions(meta)

    with open(out_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)

    print(f"Caption saved to {out_path}")
