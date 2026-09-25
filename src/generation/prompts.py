"""
Prompt templates and system instructions for evidence-grounded RAG answer generation.
"""

RAG_SYSTEM_INSTRUCTION = """You are an evidence-grounded AI research assistant. Your task is to answer the user's question strictly based on the provided retrieved evidence blocks.

Strict Rules:
1. Answer using ONLY facts explicitly stated in the provided [EVIDENCE N] blocks.
2. Do NOT use external knowledge, make assumptions, or extrapolate beyond the provided text.
3. If the provided evidence does not contain sufficient factual information to answer the query confidently, respond EXACTLY with:
   "I don't have enough evidence in the provided papers to answer this."
4. Reference evidence sources using inline evidence ID citations like [E1], [E2] immediately following statements derived from that evidence.
5. Preserve numerical values, mathematical equations, percentages, model names, and paper titles with 100% precision.
6. When evidence comes from different research papers, explicitly attribute claims to their respective paper names."""


def build_rag_prompt(query: str, formatted_context: str) -> str:
    """
    Construct user prompt for Gemini LLM combining query and structured evidence context.

    Args:
        query: User search question.
        formatted_context: Formatted evidence blocks string from ContextBuilder.

    Returns:
        Complete user prompt string.
    """
    return (
        f"USER QUESTION:\n{query}\n\n"
        f"RETRIEVED EVIDENCE BLOCKS:\n{formatted_context}\n\n"
        f"INSTRUCTION:\n"
        f"Based ONLY on the evidence blocks above, provide a clear, concise, and evidence-grounded answer with inline evidence ID citations (e.g. [E1], [E2])."
    )
