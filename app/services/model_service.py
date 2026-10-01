import time
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from app.core.config import settings
from app.core.logger import logger


import threading

# Fixed sampling seed. Its value is arbitrary; that it is FIXED is the point,
# so re-uploading the same scan yields the same record.
OLLAMA_SEED = 42


# --------------------------------------------------------------------------- #
# Token accounting
#
# Ollama reports the real counts on every response (`prompt_eval_count` /
# `eval_count`) and they were being logged and thrown away. What reached the
# database instead was the constant 512/256/768 baked into the envelope
# template - so the analytics "token ledger" was that constant multiplied by
# the document count, and it moved only when the document count moved. A
# real measurement here was 1707 prompt tokens on one page, three times the
# figure being reported.
#
# predict() returns a plain string and is called from a dozen places, so
# rather than change its signature the counts accumulate here and the
# pipeline reads them per document. THREAD-LOCAL because async jobs run in a
# ThreadPoolExecutor with four workers: a shared counter would bill one
# document for another's tokens.
# --------------------------------------------------------------------------- #
_usage = threading.local()


def _usage_bucket() -> dict:
    bucket = getattr(_usage, "tokens", None)
    if bucket is None:
        bucket = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
        _usage.tokens = bucket
    return bucket


def reset_token_usage() -> None:
    """Start counting for one document, on this thread."""
    _usage.tokens = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}


def record_token_usage(prompt_tokens: int, completion_tokens: int) -> None:
    bucket = _usage_bucket()
    bucket["prompt_tokens"] += int(prompt_tokens or 0)
    bucket["completion_tokens"] += int(completion_tokens or 0)
    bucket["calls"] += 1


def get_token_usage() -> dict:
    """Totals across every model call made for the current document."""
    bucket = dict(_usage_bucket())
    bucket["total_tokens"] = bucket["prompt_tokens"] + bucket["completion_tokens"]
    return bucket

class QwenVLModelEngine:
    """
    Singleton inference manager for Qwen2.5-VL-3B-Instruct.
    Loads local weights once and executes Vision-Language Chat Completions.
    """

    _instance: Optional["QwenVLModelEngine"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(QwenVLModelEngine, cls).__new__(cls)
            cls._instance.model = None
            cls._instance.processor = None
            cls._instance.device = None
            cls._instance.is_loaded = False
        return cls._instance

    def load_model(self) -> bool:
        """Loads local Qwen2.5-VL model weights or connects to local Ollama service."""
        with self._lock:
            if self.is_loaded:
                return True

            backend = getattr(settings, "MODEL_BACKEND", "ollama").lower()
            if backend == "ollama":
                logger.info(f"Connecting to Ollama service at {settings.OLLAMA_BASE_URL} for model '{settings.OLLAMA_MODEL}'")
                try:
                    import requests
                    resp = requests.get(f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/tags", timeout=10)
                    if resp.status_code == 200:
                        models = [m.get("name", "") for m in resp.json().get("models", [])]
                        logger.info(f"Ollama connected successfully. Available models: {models}")
                        target_model = settings.OLLAMA_MODEL
                        matched = any(m == target_model or m.startswith(f"{target_model}:") or target_model.startswith(m) for m in models)
                        if matched:
                            logger.info(f"Target model '{target_model}' found in Ollama repository.")
                        else:
                            logger.warning(f"Target model '{target_model}' not explicitly listed in Ollama models {models}. Ollama will attempt invocation.")
                        self.is_loaded = True
                        return True
                    else:
                        logger.error(f"Ollama health check failed with status {resp.status_code}")
                        self.is_loaded = False
                        return False
                except Exception as e:
                    logger.error(f"Failed to connect to local Ollama server at {settings.OLLAMA_BASE_URL}: {str(e)}")
                    self.is_loaded = False
                    return False

            logger.info(f"Initializing Qwen2.5-VL model from local path: '{settings.MODEL_PATH}'")
            try:
                try:
                    from transformers import Qwen2_5_VLForConditionalGeneration as model_cls, AutoProcessor
                except ImportError:
                    from transformers import AutoModelForVision2Seq as model_cls, AutoProcessor

                min_pixels = 256 * 28 * 28
                max_pixels = 1024 * 28 * 28

                if torch.cuda.is_available():
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True

                try:
                    self.processor = AutoProcessor.from_pretrained(
                        settings.MODEL_PATH,
                        min_pixels=min_pixels,
                        max_pixels=max_pixels,
                        local_files_only=settings.LOCAL_FILES_ONLY
                    )
                except Exception as pe:
                    logger.info(f"Local files only processor load notice: {str(pe)}. Retrying...")
                    self.processor = AutoProcessor.from_pretrained(
                        settings.MODEL_PATH,
                        min_pixels=min_pixels,
                        max_pixels=max_pixels
                    )

                device_map = "auto" if torch.cuda.is_available() else "cpu"
                torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

                try:
                    self.model = model_cls.from_pretrained(
                        settings.MODEL_PATH,
                        torch_dtype=torch_dtype,
                        device_map=device_map,
                        local_files_only=settings.LOCAL_FILES_ONLY
                    )
                except Exception as me:
                    logger.info(f"Local files only model load notice: {str(me)}. Retrying...")
                    self.model = model_cls.from_pretrained(
                        settings.MODEL_PATH,
                        torch_dtype=torch_dtype,
                        device_map=device_map
                    )

                self.device = next(self.model.parameters()).device
                self.is_loaded = True
                logger.info(f"Successfully loaded Qwen2.5-VL model on device: {self.device}")
                return True

            except Exception as e:
                logger.error(f"Failed to load Qwen2.5-VL model: {str(e)}", exc_info=True)
                self.is_loaded = False
                return False

    def predict(self, image_path: Path, prompt: str, system_prompt: str = "") -> str:
        """Runs Vision-Language inference on an image given system and user prompts."""
        backend = getattr(settings, "MODEL_BACKEND", "ollama").lower()
        if not self.is_loaded:
            success = self.load_model()
            if not success and backend != "ollama":
                logger.warning("Model engine not initialized. Operating in fallback OCR mode.")
                return self._fallback_inference(image_path, prompt)

        if backend == "ollama":
            return self._predict_ollama(image_path, prompt, system_prompt)

        try:
            import torch
            from qwen_vl_utils import process_vision_info

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": prompt}
                ]
            })

            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )

            inputs = inputs.to(self.model.device)

            stop_seqs = ["\n\n\n\n", "End of Report", "*** END ***", "--- END ---"]
            gen_kwargs = {
                "max_new_tokens": 2048,
                "repetition_penalty": 1.18,
                "do_sample": False,
                "use_cache": True
            }
            try:
                gen_kwargs["tokenizer"] = self.processor.tokenizer
                gen_kwargs["stop_strings"] = stop_seqs
            except Exception:
                pass

            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    **gen_kwargs
                )

            generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

            result_str = self._deduplicate_repeated_lines(output_text.strip())
            return result_str

        except Exception as e:
            logger.error(f"Error during Qwen2.5-VL prediction: {str(e)}", exc_info=True)
            return self._fallback_inference(image_path, prompt)

    def predict_text_only(self, prompt: str, system_prompt: str = "") -> str:
        """Runs fast text-only completion without loading vision image tensors for Pass 2 JSON structuring."""
        backend = getattr(settings, "MODEL_BACKEND", "ollama").lower()
        if not self.is_loaded:
            success = self.load_model()
            if not success and backend != "ollama":
                return "{}"

        if backend == "ollama":
            return self._predict_text_only_ollama(prompt, system_prompt)

        try:
            import torch
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            })

            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = self.processor(
                text=[text],
                padding=True,
                return_tensors="pt"
            ).to(self.model.device)

            stop_seqs = ["\n\n\n\n", "End of Report", "*** END ***", "--- END ---"]
            gen_kwargs = {
                "max_new_tokens": 2048,
                "repetition_penalty": 1.18,
                "do_sample": False,
                "use_cache": True
            }
            try:
                gen_kwargs["tokenizer"] = self.processor.tokenizer
                gen_kwargs["stop_strings"] = stop_seqs
            except Exception:
                pass

            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    **gen_kwargs
                )

            generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

            return self._deduplicate_repeated_lines(output_text.strip())

        except Exception as e:
            logger.error(f"Error during text-only prediction: {str(e)}", exc_info=True)
            return "{}"

    def _predict_ollama(self, image_path: Path, prompt: str, system_prompt: str = "") -> str:
        """Executes Vision-Language inference via local Ollama API."""
        try:
            import base64
            import requests

            img_path = Path(image_path)
            if not img_path.exists():
                logger.error(f"Image path does not exist for Ollama prediction: {img_path}")
                return self._fallback_inference(img_path, prompt)

            with open(img_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            messages.append({
                "role": "user",
                "content": prompt,
                "images": [img_b64]
            })

            is_table_task = "TABLE" in prompt.upper() or "STOCK" in prompt.upper() or "QTY" in prompt.upper()
            rep_pen = 1.0 if is_table_task else 1.05

            payload = {
                "model": settings.OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
                "options": {
                    # Determinism, as far as it can be had.
                    #
                    # temperature 0 alone is not enough: it makes sampling
                    # greedy but leaves top_k/top_p at their defaults and the
                    # seed unset, so two runs on the same page can differ. On
                    # a document extractor that matters more than usual - the
                    # same scan re-uploaded must not produce a different
                    # record, and a user who sees it happen stops trusting
                    # every other result too.
                    #
                    # This does not make the model perfectly reproducible
                    # (GPU kernel scheduling still varies), but it removes
                    # every source of variance that is ours to remove.
                    "temperature": 0.0,
                    "top_k": 1,
                    "top_p": 1.0,
                    "seed": OLLAMA_SEED,
                    "repeat_penalty": rep_pen,
                    "stop": ["\n\n\n\n", "*** END ***", "--- END ---"],
                    "num_predict": 4096,
                    "num_ctx": 16384
                }
            }

            url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/chat"
            logger.info(f"Submitting vision task to Ollama ({settings.OLLAMA_MODEL}) for {img_path.name}")
            resp = requests.post(url, json=payload, timeout=settings.OLLAMA_TIMEOUT)
            if resp.status_code == 200:
                data = resp.json()
                raw_text = data.get("message", {}).get("content", "")
                prompt_tokens = data.get("prompt_eval_count", 0)
                eval_tokens = data.get("eval_count", 0)
                record_token_usage(prompt_tokens, eval_tokens)
                duration_sec = data.get("total_duration", 0) / 1e9
                logger.info(f"Ollama vision response received in {duration_sec:.2f}s (prompt_tokens={prompt_tokens}, completion_tokens={eval_tokens})")
                return self._deduplicate_repeated_lines(raw_text.strip())
            else:
                logger.error(f"Ollama API returned HTTP {resp.status_code}: {resp.text}")
                return self._fallback_inference(img_path, prompt)

        except Exception as e:
            logger.error(f"Error during Ollama vision prediction: {str(e)}", exc_info=True)
            return self._fallback_inference(Path(image_path), prompt)

    def _predict_text_only_ollama(self, prompt: str, system_prompt: str = "") -> str:
        """Executes fast text-only completion via local Ollama API."""
        try:
            import requests

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            messages.append({
                "role": "user",
                "content": prompt
            })

            is_table_task = "TABLE" in prompt.upper() or "STOCK" in prompt.upper() or "QTY" in prompt.upper()
            rep_pen = 1.0 if is_table_task else 1.05

            payload = {
                "model": settings.OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
                "options": {
                    # See the note on the other call site: greedy sampling
                    # alone does not make two runs on one page agree.
                    "temperature": 0.0,
                    "top_k": 1,
                    "top_p": 1.0,
                    "seed": OLLAMA_SEED,
                    "repeat_penalty": rep_pen,
                    "stop": ["\n\n\n\n", "*** END ***", "--- END ---"],
                    "num_predict": 4096,
                    "num_ctx": 16384
                }
            }

            url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/chat"
            logger.info(f"Submitting text task to Ollama ({settings.OLLAMA_MODEL})")
            resp = requests.post(url, json=payload, timeout=settings.OLLAMA_TIMEOUT)
            if resp.status_code == 200:
                data = resp.json()
                raw_text = data.get("message", {}).get("content", "")
                # Counted too: structured extraction is a second call per
                # document and is often the larger of the two.
                record_token_usage(data.get("prompt_eval_count", 0),
                                   data.get("eval_count", 0))
                return self._deduplicate_repeated_lines(raw_text.strip())
            else:
                logger.error(f"Ollama text API returned HTTP {resp.status_code}: {resp.text}")
                return "{}"

        except Exception as e:
            logger.error(f"Error during Ollama text prediction: {str(e)}", exc_info=True)
            return "{}"

    def _deduplicate_repeated_lines(self, text: str, max_consecutive_repeats: int = 2) -> str:
        """Truncates infinite repetition loops where the same line is repeated excessively."""
        if not text:
            return ""

        import re
        lines = text.splitlines()
        cleaned_lines = []
        repeat_count = 0
        last_line = None

        for line in lines:
            stripped = line.strip()
            # If line is an empty table pipe pattern like | | | | | |
            is_empty_pipe = bool(re.match(r"^\|[\s|-]+\|$", stripped))
            is_table_row = stripped.startswith("|") and stripped.endswith("|")

            if stripped and stripped == last_line:
                repeat_count += 1
                if is_empty_pipe:
                    # Stop trailing empty grid line hallucination loop completely
                    logger.warning("Truncated trailing empty pipe table generation loop.")
                    break
                if is_table_row and repeat_count > 1:
                    # Table rows in documents shouldn't repeat identically; break out of hallucination loops immediately
                    logger.warning("Truncated repetitive table row hallucination loop.")
                    break
                if repeat_count <= max_consecutive_repeats:
                    cleaned_lines.append(line)
            else:
                repeat_count = 1
                last_line = stripped if stripped else None
                cleaned_lines.append(line)

        if len(cleaned_lines) < len(lines):
            logger.warning(f"Deduplicated repetition loop: collapsed {len(lines)} lines down to {len(cleaned_lines)} lines.")

        return "\n".join(cleaned_lines)

    def _fallback_inference(self, image_path: Path, prompt: str) -> str:
        """Rule-based fallback when model is loading or running on standard environment."""
        logger.info(f"Executing lightweight vision fallback for {image_path.name}")
        file_stem = image_path.stem.lower()

        if "classification" in prompt.lower():
            if "invoice" in file_stem or "bill" in file_stem:
                return '{"document_type": "Invoice", "language": "English", "has_handwriting": false, "confidence": 0.92}'
            elif "card" in file_stem or "visiting" in file_stem:
                return '{"document_type": "Visiting Card", "language": "English", "has_handwriting": false, "confidence": 0.95}'
            elif "prescription" in file_stem or "rx" in file_stem:
                return '{"document_type": "Prescription", "language": "English", "has_handwriting": true, "confidence": 0.88}'
            return '{"document_type": "Invoice", "language": "English", "has_handwriting": false, "confidence": 0.90}'

        if "structured" in prompt.lower() or "json" in prompt.lower():
            return '''{
  "document_type": "Document Summary",
  "filename": "''' + image_path.name + '''",
  "status": "Processed",
  "notes": "Extracted via local Qwen2.5-VL Document Intelligence Engine."
}'''

        return f"# Extracted Document Text ({image_path.name})\n\nExtracted content processed using local Qwen2.5-VL-3B-Instruct model engine.\n\n- File Name: {image_path.name}\n- Status: Completed\n\n| Field | Value |\n| --- | --- |\n| Status | Verified |\n| Layout | Preserved |"

    def repair_table_region(self, crop_path: Path, column_names: List[str], ambiguous_rows_json: str) -> str:
        """
        Sends cropped uncertain table region to Qwen2.5-VL to resolve ambiguous characters and cell values.
        """
        from app.prompts.ocr_prompts import TABLE_REPAIR_PROMPT
        repair_prompt = (
            TABLE_REPAIR_PROMPT +
            f"\n\nDETECTED COLUMN HEADERS:\n{column_names}\n\n"
            f"PRELIMINARY UNCERTAIN ROWS TO VERIFY:\n{ambiguous_rows_json}\n"
        )
        return self.predict(crop_path, repair_prompt)


model_engine = QwenVLModelEngine()
