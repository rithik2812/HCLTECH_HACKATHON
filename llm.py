import os, json
from typing import Any
from openai import OpenAI
from google import genai
from google.genai import types
from .prompts import SYSTEM_PROMPT, CATEGORY_GUIDANCE
from .schemas import DraftResponse

class LLMRouter:
    """
    Explicit provider router for Gemini, Groq and OpenRouter.
    Gemini uses google-genai; Groq and OpenRouter use OpenAI-compatible APIs.
    Fallback is opt-in and only attempted when ENABLE_LLM_FALLBACK=true.
    """
    def __init__(self):
        self.primary = os.getenv("LLM_PROVIDER", "groq").lower()
        self.enable_fallback = os.getenv("ENABLE_LLM_FALLBACK", "false").lower() == "true"
        self.fallback_order = [x.strip().lower() for x in
            os.getenv("LLM_FALLBACK_ORDER", "groq,openrouter,gemini").split(",") if x.strip()]
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
        self.max_tokens = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "1200"))

    def _call_gemini(self, prompt: str) -> str:
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not configured")
        model = os.getenv("LLM_MODEL", "gemini-2.5-flash")
        client = genai.Client(api_key=key)
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                response_mime_type="application/json",
            ),
        )
        return response.text or ""

    def _call_openai_compatible(self, provider: str, prompt: str) -> str:
        if provider == "groq":
            key = os.getenv("GROQ_API_KEY")
            base_url = "https://api.groq.com/openai/v1"
            model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        elif provider == "openrouter":
            key = os.getenv("OPENROUTER_API_KEY")
            base_url = "https://openrouter.ai/api/v1"
            model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
        else:
            raise ValueError(f"Unsupported OpenAI-compatible provider: {provider}")
        if not key:
            raise RuntimeError(f"{provider.upper()} API key is not configured")
        client = OpenAI(api_key=key, base_url=base_url)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content or ""

    def _call(self, provider: str, prompt: str) -> str:
        if provider == "gemini":
            return self._call_gemini(prompt)
        return self._call_openai_compatible(provider, prompt)

    def generate(self, user_request: str, category: str, customer_facts: dict,
                 deterministic_checks: dict, policy_context: str) -> DraftResponse:
        category_hint = CATEGORY_GUIDANCE.get(category.lower(), "Use a respectful, factual tone.")
        prompt = f"""
COMMUNICATION CATEGORY: {category}
CATEGORY GUIDANCE: {category_hint}

USER REQUEST:
{user_request}

VERIFIED CUSTOMER FACTS (JSON; may be empty):
{json.dumps(customer_facts, ensure_ascii=False, default=str)}

DETERMINISTIC CHECK RESULTS (JSON; authoritative for this workflow):
{json.dumps(deterministic_checks, ensure_ascii=False, default=str)}

RETRIEVED POLICY CONTEXT (reference only; not customer eligibility evidence):
{policy_context}

Draft a concise message only if the supplied evidence supports it. If not, produce a
safe draft/warning that states review is required. Return JSON only.
"""
        providers = [self.primary]
        if self.enable_fallback:
            providers += [p for p in self.fallback_order if p not in providers]
        errors = []
        for provider in providers:
            try:
                raw = self._call(provider, prompt)
                data = json.loads(raw)
                data.setdefault("status", "DRAFT_FOR_HUMAN_REVIEW")
                return DraftResponse.model_validate(data)
            except Exception as exc:
                errors.append(f"{provider}: {type(exc).__name__}: {exc}")
                if not self.enable_fallback:
                    break
        raise RuntimeError("LLM generation failed. " + " | ".join(errors))
