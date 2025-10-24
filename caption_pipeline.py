import json, os, random
from typing import Dict, Any, List
import google.generativeai as genai
from google.api_core.exceptions import NotFound, InvalidArgument

# Run this first on the terminal: 

# To see which metadata keys were actually considered in the final output
ALLOWED_FIELDS = {"main_components","length_m","num_piers","missing_components","notes","bridge_type"}


# Creates the propmt which is sent to LLM (generated 3 captions by default)
def build_prompt(meta: Dict[str, Any], k:int=1) -> str:
    """
    Produces: caption + 3 single-round QAs + 1 multi-round conversation.
    """
    tone = random.choice([
        "write in a factual yet vivid tone",
        "describe the object precisely but avoid redundancy",
        "use clear, professional phrasing appropriate for academic datasets"
    ])
    meta_llm = {fld: meta.get(fld) for fld in [
        "domain","bridge_type","main_components","length_m","num_piers",
        "missing_components","notes","target_tokens"
    ]}
    meta_llm["target_tokens"] = meta_llm.get("target_tokens", 80)

    return (
        "You are a PointLLM-style captioning assistant. "
        "Analyze the given 3D point cloud metadata and create a rich structured JSON response.\n\n"
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
        "- The caption should cover physical structure, composition, usage, and notable design details.\n"
        "- Avoid speculative or irrelevant information.\n"
        "- Use professional, descriptive language.\n"
        "- The Q&A pairs should focus on distinct and meaningful aspects (e.g purpose, users).\n"
        "- The multi-conversation should show logical continuity between questions.\n\n"
        f"{tone}.\n\n"
        f"METADATA_JSON:\n{json.dumps(meta_llm, ensure_ascii=False)}"
    )

# Checks whether the model's output is usable 
def _extract_text(resp):
    if not resp: return None
    try:
        if getattr(resp,"text",None): return resp.text
    except: pass
    try:
        parts = getattr(resp.candidates[0],"content",None)
        if parts and getattr(parts,"parts",None):
            buf=[getattr(p,"text",None) for p in parts.parts if getattr(p,"text",None)]
            if buf: return "".join(buf).strip()
    except: pass
    return None


def call_gemini(prompt: str) -> str:
    """
    Sends a prompt to the Gemini API, handles token budgeting, retries, and safe extraction.
    Returns the raw text (ideally valid JSON) from the model.
    """
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY environment variable is not set.")

    # Configure API
    genai.configure(api_key=key, transport="rest")

    # Two model options: fast + fallback (pro)
    models = ["models/gemini-2.5-flash", "models/gemini-2.5-pro"]

    # Disable content filtering to avoid blocked responses
    safety = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]

    last_err = None

    # Try models in order
    for model in models:
        try:
            m = genai.GenerativeModel(model)

            # --- Estimate input tokens ---
            try:
                token_info = m.count_tokens(prompt)
                in_tokens = int(getattr(token_info, "total_tokens", 0) or 0)
                print(f"Input tokens for {model}: {in_tokens}")
            except Exception:
                in_tokens = 0

            # --- Calculate safe output token budget ---
            max_out = min(2048, 8000 - in_tokens - 200) if in_tokens and in_tokens < 7000 else 1024

            # --- Define generation lambda ---
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

            # --- First attempt ---
            resp = gen(max_out)
            text = _extract_text(resp)

            # --- Retry with double output space if first attempt was empty ---
            if not (text and text.strip()):
                resp = gen(min(max_out * 2, 4096))
                text = _extract_text(resp)

            # --- If still empty, record issue; else return ---
            if text and text.strip():
                return text

            # Debug info for empty output
            finish_reason = getattr(resp.candidates[0], "finish_reason", None) if getattr(resp, "candidates", None) else None
            print(f"⚠️ No usable text from {model}. finish_reason={finish_reason} feedback={getattr(resp, 'prompt_feedback', None)}")
            last_err = RuntimeError(f"No usable text from {model}.")

        # --- Error handling for invalid model or other API issues ---
        except (NotFound, InvalidArgument) as e:
            last_err = e
            print(f"Model error for {model}: {e}")
        except Exception as e:
            last_err = e
            print(f" Unexpected error for {model}: {e}")

    # --- All models failed ---
    raise RuntimeError(f"All Gemini models failed. Last error: {last_err}")


# Checks whether the output has the mentioned features
def validate_against_metadata(caption:str, meta:Dict[str,Any]) -> List[str]:
    errs=[]
    if meta.get("length_m") is not None and ("m" not in caption): 
        errs.append("length lacks unit 'm'")
    if meta.get("num_piers") is not None and ("pier" not in caption): 
        errs.append("piers not referenced")
    return errs

# Final pipeline, where the built prompt is fed to the model
def generate_captions(meta: Dict[str, Any], provider="gemini") -> Dict[str, Any]:
    raw = call_gemini(build_prompt(meta))
    try:
        data = json.loads(raw[raw.find("{"):raw.rfind("}")+1])
    except Exception as e:
        raise RuntimeError(f"JSON parse failed: {e}. Raw preview: {raw[:600].replace(chr(10),' ')}")

    for k in ["caption", "single_conversation", "multi_conversation"]:
        if k not in data:
            raise RuntimeError(f"Missing '{k}' in model output. Got keys: {list(data.keys())}")

    # Run metadata validation
    issues = validate_against_metadata(data["caption"], meta)
    if issues:
        print(f"⚠️ Metadata validation issues: {issues}")

    return {
        "id": meta["id"],
        "domain": meta["domain"],
        "output": data,
        "metadata_used": sorted(ALLOWED_FIELDS & set(meta.keys())),
        "validation_issues": issues,  
    }



if __name__=="__main__":
    with open("metadata_examples.jsonl","r",encoding="utf-8") as f:
        meta = json.loads(f.readline().strip())
    out_path = "output.json"  
    with open(out_path, "w", encoding="utf-8") as file:
        json.dump(generate_captions(meta), file, indent=2, ensure_ascii=False)
    print(f"The file is saved to {out_path}")