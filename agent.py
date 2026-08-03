#!/usr/bin/env python3
"""
A simple command-line chat agent powered by the Groq API.

Setup:
    pip install groq
    export API_KEY="your-groq-api-key"   # get one at https://console.groq.com/keys
    python groq_agent.py
"""

import os
import sys

from groq import Groq, APIError, APIConnectionError, RateLimitError

MODEL = "llama-3.3-70b-versatile"
SYSTEM_PROMPT = "You are a helpful and concise AI assistant."
MAX_HISTORY_MESSAGES = 20  # keep the last N messages (excluding system prompt) to limit token usage


def get_client() -> Groq:
    """Create and return a Groq client, exiting cleanly if no API key is set."""
    api_key = os.environ.get("API_KEY")
    if not api_key:
        print("Error: set the API_KEY environment variable first.")
        print('  export API_KEY="your-groq-api-key"')
        sys.exit(1)
    return Groq(api_key=api_key)


def trim_history(messages: list[dict]) -> list[dict]:
    """Keep the system prompt plus only the most recent messages, to avoid unbounded growth."""
    system = messages[0]
    rest = messages[1:]
    if len(rest) > MAX_HISTORY_MESSAGES:
        rest = rest[-MAX_HISTORY_MESSAGES:]
    return [system] + rest


def get_reply(client: Groq, messages: list[dict]) -> str | None:
    """Call the Groq API and return the assistant's reply, or None on failure."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
        )
        return response.choices[0].message.content
    except RateLimitError:
        print("Rate limit hit — wait a moment and try again.")
    except APIConnectionError:
        print("Connection error — check your internet connection.")
    except APIError as e:
        print(f"Groq API error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    return None


def main():
    client = get_client()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("Groq Agent ready. Type 'quit' or 'exit' to leave, 'reset' to clear history.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if user_input.lower() == "reset":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("History cleared.\n")
            continue
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        messages = trim_history(messages)

        reply = get_reply(client, messages)

        if reply is None:
            # Drop the last user message so a failed call doesn't pollute history
            messages.pop()
            continue

        print(f"\nAgent: {reply}\n")
        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()