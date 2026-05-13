"""MkDocs hook: renders status badges from docs:meta frontmatter."""

import re
import os

STATUS_COLORS = {
    "Production": "success",
    "Beta": "warning",
    "Planned": "info",
}


def on_page_markdown(markdown, page, config, files, **kwargs):
    if not page.file.src_path.endswith(".md"):
        return markdown

    full_path = os.path.join(config["docs_dir"], page.file.src_path)
    if not os.path.exists(full_path):
        return markdown

    with open(full_path) as f:
        content = f.read()

    status_match = re.search(r"Status:\s*(\w+)", content)
    date_match = re.search(r"Last validated:\s*([\d-]+)", content)
    source_match = re.search(r"Source:\s*`([^`]+)`", content)

    if status_match:
        status = status_match.group(1)
        color = STATUS_COLORS.get(status, "default")
        date = date_match.group(1) if date_match else "unknown"
        source = source_match.group(1) if source_match else "N/A"

        badge = (
            f'<p style="margin-bottom:1em;">'
            f'<span class="md-tag md-tag--{color}">{status}</span> '
            f"<small>Last validated: {date} | Source: <code>{source}</code></small>"
            f"</p>"
        )
        if markdown.startswith("> **Status**"):
            markdown = re.sub(
                r"^> \*\*Status\*\*.*?\n+",
                badge + "\n",
                markdown,
                count=1,
            )

    return markdown
