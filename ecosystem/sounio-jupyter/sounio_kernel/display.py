"""Display utilities - formats epistemic values with uncertainty visualization."""

from typing import Optional


def format_knowledge_html(
    value: float,
    epsilon: float,
    provenance: str,
) -> str:
    """
    Format a Knowledge value as colored HTML for Jupyter display.

    Color scheme:
    - Green (#2ecc71):  epsilon >= 0.9  (high confidence)
    - Orange (#f39c12): epsilon >= 0.7  (medium confidence)
    - Red (#e74c3c):    epsilon < 0.7   (low confidence)

    Args:
        value: The epistemic value
        epsilon: Confidence level (0.0-1.0)
        provenance: Source/method of the value

    Returns:
        HTML string for display in Jupyter
    """
    # Determine color based on confidence level
    if epsilon >= 0.9:
        color = "#2ecc71"  # Green
        confidence_label = "High"
    elif epsilon >= 0.7:
        color = "#f39c12"  # Orange
        confidence_label = "Medium"
    else:
        color = "#e74c3c"  # Red
        confidence_label = "Low"

    # Format confidence percentage
    confidence_percent = int(epsilon * 100)

    # Build HTML with inline CSS
    html = f"""<div class="epistemic-value" style="
        border-left: 4px solid {color};
        padding-left: 10px;
        padding: 8px;
        background-color: rgba(0, 0, 0, 0.02);
        border-radius: 4px;
        font-family: monospace;
        margin: 4px 0;
    ">
        <div style="font-weight: bold; color: {color};">
            Value: {value:.6g}
        </div>
        <div style="font-size: 0.9em; color: #555; margin-top: 4px;">
            <span style="color: {color};">
                ●
            </span>
            Confidence: <span style="color: {color}; font-weight: 600;">
                {confidence_percent}%
            </span>
            ({confidence_label}) | Provenance: <code style="background-color: rgba(0,0,0,0.05); padding: 2px 4px; border-radius: 3px;">{provenance}</code>
        </div>
    </div>"""

    return html


def format_knowledge_text(
    value: float,
    epsilon: float,
    provenance: str,
) -> str:
    """
    Format a Knowledge value as plain text.

    Args:
        value: The epistemic value
        epsilon: Confidence level (0.0-1.0)
        provenance: Source/method of the value

    Returns:
        Plain text string representation
    """
    confidence_percent = int(epsilon * 100)
    return f"Knowledge {{ value: {value:.6g}, ε: {confidence_percent}%, prov: {provenance} }}"


def colorize_confidence(epsilon: float) -> str:
    """
    Get ANSI color code for confidence level (for terminal output).

    Args:
        epsilon: Confidence level (0.0-1.0)

    Returns:
        ANSI escape sequence for color
    """
    if epsilon >= 0.9:
        return "\033[92m"  # Green
    elif epsilon >= 0.7:
        return "\033[93m"  # Yellow/Orange
    else:
        return "\033[91m"  # Red


def reset_ansi_color() -> str:
    """Return ANSI reset sequence."""
    return "\033[0m"
