"""Generate natural-language answers to bylaw questions from retrieved
passages, grounded only in that retrieved text.
"""

from anthropic import Anthropic

from src.retriever import SearchResult

MODEL_NAME = "claude-opus-5"

ANSWER_PROMPT = """You are answering a resident's question about City of Calgary bylaws.

Question: {question}

Below are the most relevant excerpts retrieved from the bylaws. Answer \
using ONLY the information in these excerpts -- do not use outside \
knowledge of the bylaws.

Rules:
- If the excerpts do not contain enough information to answer, say so \
plainly instead of guessing.
- Cite the bylaw name and section for each claim you make, e.g. \
"(Parking Bylaw, Section 3)".
- Keep the answer concise: a few sentences, plain language, no legal \
jargon beyond what's necessary.

Excerpts:
{excerpts}
"""


def format_excerpts(results: list[SearchResult]) -> str:
    blocks = []
    for r in results:
        section = f"Section {r.section}" if r.section else "Preamble"
        blocks.append(f"[{r.bylaw_name} -- {section}, page {r.page}]\n{r.text}")
    return "\n\n---\n\n".join(blocks)


def generate_answer(client: Anthropic, question: str, results: list[SearchResult]) -> str:
    """Ask Claude to answer the question using only the retrieved passages."""
    if not results:
        return "I couldn't find any relevant bylaw text to answer that question."

    prompt = ANSWER_PROMPT.format(question=question, excerpts=format_excerpts(results))
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    # Claude Opus 5 thinks by default, so content[0] may be a ThinkingBlock --
    # find the actual text block rather than assuming it's first.
    text = next((block.text for block in response.content if block.type == "text"), "")
    return text.strip()
