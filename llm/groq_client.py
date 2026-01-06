import os
from groq import Groq


# Create Groq client using Streamlit Secrets / environment variables
api_key = os.environ.get("GROQ_API_KEY")

if not api_key:
    raise RuntimeError(
        "GROQ_API_KEY not found. Please add it in Streamlit Secrets."
    )

client = Groq(api_key=api_key)


def call_llm(prompt):
    response = client.chat.completions.create(
        model="qwen/qwen3-32b",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content
