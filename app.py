import streamlit as st
import requests
import json
import re
import torch
from collections import defaultdict

import os
os.environ["STREAMLIT_WATCHDOG_MODE"] = "poll"

# Device info
device = "cuda" if torch.cuda.is_available() else "cpu"

# Streamlit config
st.set_page_config(page_title="HumanizeText", layout="centered")
st.title("HumanizeText")

# API Config
API_URL = "http://localhost:3000/api/chat/completions"
API_KEY = ""

# System Prompt
SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You're a real human content writer, not an AI. Rewrite the text to sound like a casual human wrote it in a hurry, based on their real experience.\n"
        "Use contractions, incomplete thoughts, and natural language. Vary sentence length. Break grammar rules slightly.\n"
        "Add a small personal anecdote or opinion, even if it's just a line. Use simple, conversational words.\n"
        "Avoid sounding perfect. No fluff. No AI tone. No corporate-speak.\n"
        "It should read like a human typed it up quickly, honestly, and naturally.\n"
        "Your goal: 0% AI detection. Return only the humanized version."
    )
}

# Function to get response + references
def get_response(messages):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "gpt-4o",
        "messages": messages,
        "temperature": 0.8,
        "top_p": 1.0
    }

    response = requests.post(API_URL, headers=headers, data=json.dumps(data))

    if response.status_code != 200:
        return f"❌ Error: {response.status_code}, {response.text}"

    json_resp = response.json()
    content = json_resp.get("choices", [{}])[0].get("message", {}).get("content", "No response")
    tool_calls = json_resp.get("choices", [{}])[0].get("message", {}).get("tool_calls", [])

    # Remove in-line [0], [1] citations
    content = re.sub(r'\[\d+\]', '', content)

    # Organize references
    ref_dict = defaultdict(set)

    for tool in tool_calls:
        function = tool.get("function", {})
        if function.get("name") == "retrieval":
            try:
                arguments = json.loads(function.get("arguments", "{}"))
                docs = arguments.get("documents", [])
                for doc in docs:
                    source = doc.get("source", "Unknown").strip()
                    page = doc.get("page", "").strip()
                    url = doc.get("url", "").strip()
                    ref_dict[source].add((page, url))
            except Exception as e:
                st.error(f"⚠️ Failed to parse citations: {e}")

    # Build clean reference section
    if ref_dict:
        content += "\n\n---\n### References\n"
        for idx, (title, entries) in enumerate(ref_dict.items(), 1):
            pages = sorted(set(p for p, _ in entries if p))
            url = next((u for _, u in entries if u), "")
            page_str = f"pp. {', '.join(pages)}" if len(pages) > 1 else f"p. {pages[0]}" if pages else ""
            line = f"{idx}. **{title}**"
            if page_str:
                line += f", {page_str}"
            if url:
                line += f" - [View]({url})"
            content += f"{line}\n"

    return content

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [SYSTEM_PROMPT]

# Display previous messages (excluding system)
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)

# User input
user_input = st.chat_input("Paste text to humanize:")

if user_input:
    # Count words in input
    word_count = len(user_input.strip().split())
    st.markdown(f"📏 **Word Count:** {word_count}")

    # Warn if input is too long
    if word_count > 1000:
        st.warning("⚠️ Your input is very long and might exceed the model's processing limits. Consider splitting it.")

    # Store user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get and show AI response
    with st.spinner("Retrieving..."):
        response = get_response(st.session_state.messages)

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response, unsafe_allow_html=True)
