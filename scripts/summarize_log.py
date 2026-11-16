import os
import sys
from pathlib import Path

import requests

LOG_FILE = sys.argv[1] if len(sys.argv) > 1 else "build.log"
API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-120b"


def summarize_log():
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        print("Error: LLM_API_KEY environment variable is missing.", file=sys.stderr)
        sys.exit(1)

    log_path = Path(LOG_FILE)
    if not log_path.is_file():
        print(f"Error: Log file not found: {LOG_FILE}", file=sys.stderr)
        sys.exit(1)

    # Keep the prompt within a predictable size by sending only the log tail.
    log_text = log_path.read_text(encoding="utf-8", errors="replace")[-6000:]

    prompt = f"""Analyze this CI build log and provide:
1. Root cause of the failure.
2. Failing test or file, if identifiable.
3. A practical suggested fix.

Use only evidence in the log. Do not invent filenames, line numbers,
or errors. If the cause cannot be determined, say so.

BUILD LOG:
{log_text}
"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are an expert CI/CD build-log analysis assistant.",
            },
            {"role": "user", "content": prompt},
        ],
        "max_completion_tokens": 500,
        "temperature": 0.2,
    }

    try:
        response = requests.post(
            API_URL, headers=headers, json=payload, timeout=60
        )
        response.raise_for_status()
        data = response.json()
        summary = data["choices"][0]["message"]["content"]
        print("## LLM Build Failure Summary\n")
        print(summary)
    except requests.RequestException as error:
        print(f"Groq API request failed: {error}", file=sys.stderr)
        sys.exit(1)
    except (KeyError, IndexError, TypeError, ValueError) as error:
        print(f"Could not read the LLM response: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    summarize_log()
