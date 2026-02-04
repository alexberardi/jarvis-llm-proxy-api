"""
Date key extraction service and constants.

This module defines the vocabulary of supported date keys and provides
utilities for date key extraction using a hybrid FastText + LLM approach.

Extraction strategy:
- FastText confidence >= 85%: Use FastText directly (fastest, ~1ms)
- FastText confidence 75-85%: Add FastText hint to LLM prompt
- FastText confidence < 75%: Let LLM handle independently
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

logger = logging.getLogger("uvicorn")

# ============================================================================
# DATE KEY VOCABULARY
# ============================================================================

# Version for cache invalidation - increment when keys change
DATE_KEYS_VERSION = "1.1"

# Relative days
RELATIVE_DAYS = [
    "today",
    "tomorrow",
    "yesterday",
    "day_after_tomorrow",
    "day_before_yesterday",
]

# Combined day+time standalone keys
COMBINED_KEYS = [
    "tonight",
    "last_night",
    "tomorrow_night",
    "tomorrow_morning",
    "tomorrow_afternoon",
    "tomorrow_evening",
    "yesterday_morning",
    "yesterday_afternoon",
    "yesterday_evening",
    "this_morning",
    "this_afternoon",
    "this_evening",
]

# Time of day modifiers
TIME_MODIFIERS = [
    "morning",
    "afternoon",
    "evening",
    "night",
    "noon",
    "midnight",
    "at_noon",      # "at noon" variant
    "at_midnight",  # "at midnight" variant
]

# Meal times
MEAL_TIMES = [
    "at_breakfast",
    "during_breakfast",
    "during_lunch",
    "at_dinner",
    "during_dinner",
    "after_dinner",
]

# Weekdays with prefixes
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
WEEKDAY_KEYS = []
for prefix in ["this", "next", "last"]:
    for day in WEEKDAYS:
        WEEKDAY_KEYS.append(f"{prefix}_{day}")

# Periods
PERIOD_KEYS = [
    "this_week", "next_week", "last_week",
    "this_weekend", "next_weekend", "last_weekend",
    "this_month", "next_month", "last_month",
    "this_year", "next_year", "last_year",
]

# All static keys (time patterns are dynamic)
ALL_DATE_KEYS = (
    RELATIVE_DAYS +
    COMBINED_KEYS +
    TIME_MODIFIERS +
    MEAL_TIMES +
    WEEKDAY_KEYS +
    PERIOD_KEYS
)

# Time patterns documentation
TIME_PATTERNS = {
    "hourly": "at_Xam, at_Xpm (X = 1-12)",
    "quarter_past": "at_X_15am, at_X_15pm",
    "half_past": "at_X_30am, at_X_30pm",
    "quarter_to": "at_X_45am, at_X_45pm",
}

# Confidence thresholds for hybrid extraction
FASTTEXT_HIGH_CONFIDENCE = 0.85  # Use FastText directly
FASTTEXT_HINT_THRESHOLD = 0.75   # Add FastText hint to LLM


def get_date_keys_response() -> dict:
    """
    Get the full date keys vocabulary response.

    Returns the structure expected by /v1/adapters/date-keys endpoint.
    """
    return {
        "version": DATE_KEYS_VERSION,
        "keys": sorted(ALL_DATE_KEYS),
        "patterns": TIME_PATTERNS,
        "notes": {
            "composability": "Multiple keys may be returned, e.g., ['next_tuesday', 'morning']",
            "no_date": "Empty array returned if no date reference detected",
            "combined_keys": "Common phrases like 'tonight', 'tomorrow_morning' are standalone keys",
        }
    }


def get_adapter_path() -> Optional[Path]:
    """
    Get the path to the trained date key adapter if it exists.

    Returns None if no adapter is trained yet.
    """
    adapter_path = Path("adapters/date_keys")
    if adapter_path.exists() and (adapter_path / "adapter_config.json").exists():
        return adapter_path
    return None


def get_fasttext_model_path() -> Optional[Path]:
    """
    Get the path to the FastText date key model if it exists.
    """
    model_path = Path("models/date_keys_fasttext.bin")
    if model_path.exists():
        return model_path
    return None


def is_adapter_trained() -> bool:
    """Check if the date key adapter has been trained."""
    return get_adapter_path() is not None


# ============================================================================
# FASTTEXT DATE KEY EXTRACTOR (Fast, lightweight)
# ============================================================================

@dataclass
class FastTextPrediction:
    """Result from FastText date key extraction."""
    keys: List[str]
    max_confidence: float
    all_predictions: List[Tuple[str, float]]  # (key, score) pairs


class FastTextDateKeyExtractor:
    """
    Fast date key extraction using FastText classifier.

    ~1ms inference on CPU, 5-10MB model size.
    """

    _instance: Optional["FastTextDateKeyExtractor"] = None
    _model = None
    _loaded = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _ensure_loaded(self) -> bool:
        """Load the FastText model if not already loaded."""
        if self._loaded:
            return True

        model_path = get_fasttext_model_path()
        if not model_path:
            return False

        try:
            # Patch numpy for compatibility with numpy 2.0
            import numpy as np
            _old_array = np.array
            def _new_array(*args, **kwargs):
                if 'copy' in kwargs and kwargs['copy'] is False:
                    kwargs['copy'] = None
                return _old_array(*args, **kwargs)
            np.array = _new_array

            import fasttext
            self._model = fasttext.load_model(str(model_path))
            self._loaded = True
            logger.info(f"✅ FastText date key extractor loaded ({len(self._model.labels)} labels)")
            return True
        except Exception as e:
            logger.warning(f"⚠️ FastText model not available: {e}")
            return False

    def predict(self, text: str, k: int = 5, threshold: float = 0.3) -> FastTextPrediction:
        """
        Predict date keys from text.

        Args:
            text: Input text to analyze
            k: Max number of labels to return
            threshold: Minimum confidence to include a label

        Returns:
            FastTextPrediction with keys, max confidence, and all predictions
        """
        if not self._ensure_loaded():
            return FastTextPrediction(keys=[], max_confidence=0.0, all_predictions=[])

        try:
            text_clean = text.lower().strip()
            if not text_clean:
                return FastTextPrediction(keys=[], max_confidence=0.0, all_predictions=[])

            labels, scores = self._model.predict(text_clean, k=k)

            keys = []
            all_preds = []
            max_conf = 0.0

            for label, score in zip(labels, list(scores)):
                key = label.replace("__label__", "")
                if key != "NONE":
                    all_preds.append((key, float(score)))
                    if score >= threshold:
                        keys.append(key)
                        max_conf = max(max_conf, score)

            return FastTextPrediction(
                keys=keys,
                max_confidence=max_conf,
                all_predictions=all_preds
            )
        except Exception as e:
            logger.warning(f"⚠️ FastText prediction error: {e}")
            return FastTextPrediction(keys=[], max_confidence=0.0, all_predictions=[])


# ============================================================================
# LLM DATE KEY EXTRACTOR (Accurate, slower)
# ============================================================================

class LLMDateKeyExtractor:
    """
    Extracts date keys from text using the trained LoRA adapter.

    Uses lazy loading to avoid startup delays. The model is loaded
    with 4-bit quantization to fit alongside vLLM.
    """

    _instance: Optional["LLMDateKeyExtractor"] = None
    _model = None
    _tokenizer = None
    _loaded = False
    _device = "cpu"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _ensure_loaded(self) -> bool:
        """Load the model if not already loaded. Returns True if loaded successfully."""
        if self._loaded:
            return True

        # Check if LLM extraction is disabled (e.g., when date extraction is merged into main model)
        if os.getenv("JARVIS_DISABLE_DATE_KEY_LLM", "").lower() in ("1", "true", "yes"):
            logger.info("ℹ️  LLM date key extraction disabled via JARVIS_DISABLE_DATE_KEY_LLM")
            return False

        adapter_path = get_adapter_path()
        if not adapter_path:
            logger.warning("⚠️  Date key adapter not trained yet, skipping LLM extraction")
            return False

        # Use lightweight model env var, or fall back to Llama 1B
        local_model_path = os.getenv("JARVIS_LIGHTWEIGHT_MODEL_NAME", ".models/llama-3.2-1b-instruct")
        if os.path.exists(local_model_path):
            base_model = local_model_path
        else:
            base_model = "meta-llama/Llama-3.2-1B-Instruct"

        try:
            logger.info(f"📦 Loading LLM date key extractor (base: {base_model}, adapter: {adapter_path})...")

            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            from peft import PeftModel

            # Default to CPU to avoid GPU memory contention with vLLM
            device_map = os.getenv("JARVIS_DATE_KEY_DEVICE_MAP", "cpu")
            use_quantization = device_map != "cpu"

            self._tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
            self._tokenizer.pad_token = self._tokenizer.eos_token

            if use_quantization:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                max_memory = {0: "2GiB", "cpu": "8GiB"}
                base = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    quantization_config=bnb_config,
                    device_map="auto",
                    max_memory=max_memory,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                )
            else:
                logger.info(f"🖥️  Loading LLM date key extractor on CPU")
                base = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    device_map="cpu",
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                )

            self._model = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=False)
            self._model.eval()

            try:
                self._device = next(self._model.parameters()).device
            except StopIteration:
                self._device = torch.device("cpu")

            self._loaded = True
            logger.info(f"✅ LLM date key extractor loaded successfully on {self._device}")
            return True

        except Exception as e:
            logger.exception(f"❌ Failed to load LLM date key extractor: {e}")
            return False

    # System prompt used during training - must match exactly
    _SYSTEM_PROMPT = """You are a date/time extraction assistant. Extract date and time references from the user's text and return them as a JSON array of semantic keys.

Rules:
- Return only the JSON array, nothing else
- Use standardized keys like: today, tomorrow, yesterday, tomorrow_morning, tonight, last_night, next_monday, this_weekend, at_3pm, etc.
- Return [] if no date/time references are found
- Multiple keys can be returned for composite expressions like "next Tuesday at 3pm" -> ["next_tuesday", "at_3pm"]"""

    def extract(self, text: str, hint: Optional[List[str]] = None) -> List[str]:
        """
        Extract date keys from the given text.

        Args:
            text: The input text (e.g., user's voice command)
            hint: Optional FastText suggestion to include in prompt

        Returns:
            List of date keys (e.g., ["tomorrow", "morning"]) or empty list
        """
        if not self._ensure_loaded():
            return []

        try:
            import torch

            # Build user message with optional hint
            user_content = f'Extract date keys from: "{text}"'
            if hint:
                user_content += f'\n\nHint: FastText suggests {json.dumps(hint)}'

            messages = [
                {"role": "system", "content": self._SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ]
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=self._tokenizer.eos_token_id,
                )

            response = self._tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()

            try:
                date_keys = json.loads(response)
                if isinstance(date_keys, list):
                    return [k for k in date_keys if isinstance(k, str)]
            except json.JSONDecodeError:
                if response.startswith("[") and "]" in response:
                    try:
                        partial = response[:response.index("]") + 1]
                        date_keys = json.loads(partial)
                        if isinstance(date_keys, list):
                            return [k for k in date_keys if isinstance(k, str)]
                    except (json.JSONDecodeError, TypeError, ValueError):
                        pass
                logger.warning(f"⚠️  Could not parse date keys from response: {response!r}")

            return []

        except Exception as e:
            logger.error(f"❌ LLM date key extraction error: {e}")
            return []

    def unload(self):
        """Unload the model to free memory."""
        if self._model:
            del self._model
            self._model = None
        if self._tokenizer:
            del self._tokenizer
            self._tokenizer = None
        self._loaded = False

        import gc
        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        logger.info("🧹 LLM date key extractor unloaded")


# ============================================================================
# HYBRID EXTRACTOR (Main entry point)
# ============================================================================

class HybridDateKeyExtractor:
    """
    Hybrid date key extraction using FastText + LLM.

    Strategy:
    - FastText confidence >= 85%: Use FastText directly (~1ms)
    - FastText confidence 75-85%: Add FastText hint to LLM prompt
    - FastText confidence < 75%: Let LLM handle independently
    """

    _instance: Optional["HybridDateKeyExtractor"] = None
    _fasttext: Optional[FastTextDateKeyExtractor] = None
    _llm: Optional[LLMDateKeyExtractor] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._fasttext = FastTextDateKeyExtractor()
            cls._instance._llm = LLMDateKeyExtractor()
        return cls._instance

    def extract(self, text: str) -> List[str]:
        """
        Extract date keys using hybrid FastText + LLM approach.

        Args:
            text: Input text to extract date keys from

        Returns:
            List of date keys, or empty list if no dates found
        """
        # Step 1: Try FastText (fast)
        ft_result = self._fasttext.predict(text)

        # High confidence - use FastText directly
        if ft_result.max_confidence >= FASTTEXT_HIGH_CONFIDENCE:
            logger.debug(f"⚡ FastText high confidence ({ft_result.max_confidence:.2f}): {ft_result.keys}")
            return ft_result.keys

        # No FastText model or very low confidence - LLM only
        if ft_result.max_confidence < FASTTEXT_HINT_THRESHOLD:
            if ft_result.max_confidence > 0:
                logger.debug(f"🤖 FastText low confidence ({ft_result.max_confidence:.2f}), using LLM only")
            return self._llm.extract(text, hint=None)

        # Medium confidence - give hint to LLM
        logger.debug(f"💡 FastText medium confidence ({ft_result.max_confidence:.2f}), hinting LLM with: {ft_result.keys}")
        return self._llm.extract(text, hint=ft_result.keys)

    def unload(self):
        """Unload all models."""
        if self._llm:
            self._llm.unload()


# Module-level singleton
_extractor: Optional[HybridDateKeyExtractor] = None


def extract_date_keys(text: str) -> List[str]:
    """
    Extract date keys from text using hybrid FastText + LLM approach.

    This is the main entry point for date key extraction.

    Args:
        text: Input text to extract date keys from

    Returns:
        List of date keys, or empty list if extraction fails/no dates found
    """
    global _extractor
    if _extractor is None:
        _extractor = HybridDateKeyExtractor()
    return _extractor.extract(text)


def unload_date_key_extractor():
    """Unload the date key extractor to free memory."""
    global _extractor
    if _extractor:
        _extractor.unload()
        _extractor = None
