#!/usr/bin/env python3
"""
vLLM Client for text generation.
Connects to the vLLM server started by vllm_serve.sh
"""

from openai import OpenAI

# Server configuration (matches vllm_serve.sh defaults)
HOST = "localhost"  # Change to server IP if running remotely
PORT = 9000         # Default port from vllm_serve.sh
MODEL = "openai/gpt-oss-120b"  # Default model from vllm_serve.sh

# Initialize OpenAI client pointing to vLLM server
client = OpenAI(
    base_url=f"http://{HOST}:{PORT}/v1",
    api_key="EMPTY"  # vLLM doesn't require an API key
)

def generate_text(prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
    """
    Generate text using the vLLM server.

    Args:
        prompt: The input prompt for text generation
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (0.0 to 1.0)

    Returns:
        Generated text string
    """
    response = client.completions.create(
        model=MODEL,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    return response.choices[0].text


if __name__ == "__main__":
    # Example prompt
    prompt = "Once upon a time in a distant galaxy,"

    print(f"Prompt: {prompt}")
    print("-" * 80)

    # Generate and print text
    generated_text = generate_text(prompt, max_tokens=4096, temperature=1)
    print(f"Generated text:\n{generated_text}")
