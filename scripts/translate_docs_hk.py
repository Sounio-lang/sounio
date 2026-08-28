import os
import glob
from pathlib import Path
from openai import OpenAI

# ==========================================
# Configuration for Local LLM
# ==========================================
# Adjust these to match your local LLM setup (Ollama, vLLM, LM Studio, etc.)
LOCAL_API_BASE = os.environ.get("OPENAI_API_BASE", "http://localhost:11434/v1") # Default: Ollama
LOCAL_API_KEY = os.environ.get("OPENAI_API_KEY", "not-needed-for-local")
MODEL_NAME = os.environ.get("MODEL_NAME", "llama3") # Change to your local model name

# Paths
SOURCE_DIR = Path("website/src/content/docs/en")
TARGET_DIR = Path("website/src/content/docs/zh-hk")

# Initialize OpenAI client pointed to local server
client = OpenAI(
    base_url=LOCAL_API_BASE,
    api_key=LOCAL_API_KEY,
)

SYSTEM_PROMPT = """You are an expert technical translator specializing in Hong Kong Cantonese.
Your task is to translate the provided Markdown/MDX documentation from English to authentic Hong Kong Cantonese (Traditional Chinese).

CRITICAL RULES:
1. Use authentic Hong Kong Cantonese vocabulary and phrasing (e.g., 用, 喺, 嘅, 呢個, 唔, 咗), but keep it professional enough for software documentation.
2. PRESERVE ALL Markdown formatting, including headers, lists, links, and bold/italic text.
3. PRESERVE ALL MDX components (e.g., <Card>, <Note>) exactly as they are.
4. PRESERVE ALL frontmatter (the YAML block at the very top between --- and ---). Only translate the `title` and `description` values. Do NOT translate keys like `topic_id` or `category`.
5. DO NOT translate any code blocks, terminal commands, function names, or inline code snippets (`code`).
6. Return ONLY the translated document, no conversational preamble or postamble.
"""

def translate_content(content: str) -> str:
    """Send content to local LLM for translation."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content}
            ],
            temperature=0.3, # Low temperature for more consistent translations
            max_tokens=2000,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error during translation: {e}")
        return None

def main():
    print(f"Starting HK Cantonese translation script...")
    print(f"Connecting to Local LLM at {LOCAL_API_BASE} using model '{MODEL_NAME}'")
    
    # Ensure target directory exists
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find all english MDX files
    source_files = list(SOURCE_DIR.rglob("*.mdx"))
    print(f"Found {len(source_files)} files to process.")
    
    for source_path in source_files:
        # Calculate target path
        rel_path = source_path.relative_to(SOURCE_DIR)
        target_path = TARGET_DIR / rel_path
        
        # Skip if already exists (e.g., our compiler.mdx proof of concept)
        if target_path.exists():
            print(f"Skipping {rel_path} (already exists)")
            continue
            
        print(f"Translating: {rel_path}...")
        
        # Ensure subdirectories exist
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Read source
        with open(source_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Translate
        translated_content = translate_content(content)
        
        if translated_content:
            # Write target
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(translated_content)
            print(f"  ✓ Saved to {target_path}")
        else:
            print(f"  ✗ Failed to translate {rel_path}")

if __name__ == "__main__":
    main()
