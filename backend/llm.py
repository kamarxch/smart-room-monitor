"""llm.py - one function, ask_llm(), that hides which LLM provider is used.

Choose the provider in backend/.env:
    LLM_PROVIDER=azure       (Azure OpenAI, the stack named in the PwC offer)
    LLM_PROVIDER=openai
    LLM_PROVIDER=anthropic
The rest of the project only calls ask_llm(system, user), so switching
provider never touches assistant.py or dashboard.py.
"""
import os

from dotenv import load_dotenv

load_dotenv()  # reads backend/.env (ignored by git)

PROVIDER = os.getenv("LLM_PROVIDER", "azure").lower()


def _need(name):
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing {name}. Set it in backend/.env (see .env.example).")
    return value


def ask_llm(system, user, max_tokens=600):
    """Send one system prompt + one user message, return the reply text."""
    # Thinking models (for example Gemini 3) spend part of this budget on hidden
    # reasoning, so a small limit can leave an empty answer. Replies stay short
    # anyway because the prompts ask for short answers.
    max_tokens = max(max_tokens, 1500)
    if PROVIDER == "azure":
        from openai import AzureOpenAI

        client = AzureOpenAI(
            api_key=_need("AZURE_OPENAI_API_KEY"),
            azure_endpoint=_need("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01"),
            max_retries=5, timeout=90,
        )
        resp = client.chat.completions.create(
            model=_need("AZURE_OPENAI_DEPLOYMENT"),   # the deployment name, not the model name
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        return resp.choices[0].message.content

    if PROVIDER == "openai":
        from openai import OpenAI

        # OPENAI_BASE_URL lets this same branch talk to any OpenAI-compatible
        # provider (for example Google's Gemini free tier)
        # max_retries: the SDK waits and retries temporary errors (429 rate limit,
        # 503 provider overloaded), which are common on free tiers
        client = OpenAI(api_key=_need("OPENAI_API_KEY"),
                        base_url=os.getenv("OPENAI_BASE_URL") or None,
                        max_retries=5, timeout=90)
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        return resp.choices[0].message.content

    if PROVIDER == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=_need("ANTHROPIC_API_KEY"))
        resp = client.messages.create(
            model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text

    raise RuntimeError(f"Unknown LLM_PROVIDER '{PROVIDER}' (use azure, openai or anthropic)")