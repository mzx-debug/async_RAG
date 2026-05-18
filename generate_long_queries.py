#!/usr/bin/env python3
"""
Generate a set of extra-long queries (>128 tokens) to test chunked embedding logic.

Strategy:
- Take existing queries from queries_generated.jsonl
- Expand them into multi-part compound questions with detailed context
- Target: 100 queries with 150-500 tokens each
"""

import json
import random
from pathlib import Path
from typing import List

INPUT_PATH = Path("E:/R1/async_rag_pipeline_v0/data/queries_generated.jsonl")
OUTPUT_PATH = Path("E:/R1/async_rag_pipeline_v0/data/queries_long.jsonl")

# Expansion templates to create compound questions
EXPANSION_TEMPLATES = [
    (
        "Provide a comprehensive analysis of {query} "
        "Begin by defining the core concept and its historical development from early origins to present day. "
        "Then explain the underlying mechanisms, key components, and how they interact with each other. "
        "Discuss the practical applications across different domains and industries, including specific use cases and real-world examples. "
        "Analyze the advantages and limitations, comparing it with alternative approaches or competing theories. "
        "Address common misconceptions and clarify what distinguishes it from closely related concepts. "
        "Finally, evaluate current research trends, unresolved challenges, and promising directions for future development."
    ),
    (
        "Write a detailed multi-perspective examination of {query} "
        "First, establish the foundational context by tracing its evolution through major historical milestones and paradigm shifts. "
        "Explain the theoretical framework and scientific principles that govern its behavior or operation. "
        "Describe how different stakeholders—researchers, practitioners, policymakers, and end users—perceive and interact with it. "
        "Outline the methodologies commonly used to study, measure, or implement it, including their strengths and weaknesses. "
        "Present evidence from empirical studies, case analyses, and documented outcomes to support key claims. "
        "Identify critical success factors, common pitfalls, and lessons learned from both successful and failed implementations. "
        "Conclude with actionable insights and recommendations for practitioners in the field."
    ),
    (
        "Conduct a thorough investigation into {query} covering multiple dimensions. "
        "Start with a precise definition and scope, clarifying boundaries and distinguishing it from related phenomena. "
        "Trace its intellectual lineage, identifying seminal works, key contributors, and major theoretical breakthroughs. "
        "Explain the causal mechanisms, feedback loops, and systemic interactions that characterize its dynamics. "
        "Examine its impact across different scales—individual, organizational, societal—and across different contexts. "
        "Discuss the role of technology, policy, culture, and economics in shaping its development and adoption. "
        "Analyze ongoing debates, controversies, and areas where expert consensus has not yet been reached. "
        "Synthesize findings to identify the most important takeaways and their implications for theory and practice."
    ),
    (
        "Develop a comprehensive understanding of {query} through systematic analysis. "
        "Begin with the problem statement or need that motivated its emergence, including historical antecedents. "
        "Describe the conceptual architecture, breaking down complex structures into understandable components. "
        "Explain the operational principles, workflows, or processes involved in its functioning. "
        "Evaluate performance metrics, benchmarks, and criteria used to assess effectiveness or quality. "
        "Compare and contrast different schools of thought, methodological approaches, or implementation strategies. "
        "Discuss scalability, sustainability, and adaptability considerations for different contexts and conditions. "
        "Address ethical, social, and environmental implications that practitioners and policymakers should consider. "
        "Conclude with a forward-looking perspective on emerging trends and transformative possibilities."
    ),
    (
        "Explore {query} in depth from foundational principles to cutting-edge developments. "
        "Establish the conceptual groundwork by defining key terms, assumptions, and boundary conditions. "
        "Review the historical trajectory, highlighting pivotal moments, influential figures, and paradigm shifts. "
        "Explain the underlying science, mathematics, or logic that provides the theoretical foundation. "
        "Describe practical implementations, including tools, techniques, and best practices used by experts. "
        "Analyze empirical evidence from controlled studies, field observations, and meta-analyses. "
        "Discuss interdisciplinary connections and how insights from other fields have enriched understanding. "
        "Identify knowledge gaps, methodological limitations, and areas requiring further investigation. "
        "Synthesize the analysis to provide actionable guidance for researchers, practitioners, and decision-makers."
    ),
]


def load_base_queries(path: Path) -> List[str]:
    """Load existing queries from jsonl file."""
    queries = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                queries.append(data["question"])
    return queries


def generate_long_queries(base_queries: List[str], target_count: int = 100, seed: int = 42) -> List[dict]:
    """Generate long queries by expanding base queries with detailed templates."""
    rng = random.Random(seed)

    # Sample base queries (with replacement to reach target count)
    sampled = rng.choices(base_queries, k=target_count)

    long_queries = []
    for i, base_query in enumerate(sampled):
        # Choose a random expansion template
        template = rng.choice(EXPANSION_TEMPLATES)

        # Generate the long query
        long_query = template.format(query=base_query)

        long_queries.append({
            "id": f"long_{i:04d}",
            "question": long_query,
            "base_query": base_query,
            "template_idx": EXPANSION_TEMPLATES.index(template),
        })

    return long_queries


def main():
    print(f"Loading base queries from {INPUT_PATH}...")
    base_queries = load_base_queries(INPUT_PATH)
    print(f"  Loaded {len(base_queries)} base queries")

    print("\nGenerating long queries...")
    long_queries = generate_long_queries(base_queries, target_count=100)

    # Calculate token statistics (approximate: 1 token ≈ 0.75 words)
    word_counts = [len(q["question"].split()) for q in long_queries]
    token_estimates = [int(wc * 1.33) for wc in word_counts]  # rough estimate

    print(f"\nGenerated {len(long_queries)} long queries")
    print(f"  Word count - min: {min(word_counts)}, max: {max(word_counts)}, avg: {sum(word_counts)/len(word_counts):.1f}")
    print(f"  Token estimate - min: {min(token_estimates)}, max: {max(token_estimates)}, avg: {sum(token_estimates)/len(token_estimates):.1f}")
    print(f"  Queries > 128 tokens (est): {sum(1 for t in token_estimates if t > 128)}")
    print(f"  Queries > 256 tokens (est): {sum(1 for t in token_estimates if t > 256)}")

    # Save to file
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for q in long_queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"\nSaved to {OUTPUT_PATH}")

    # Show 2 samples
    print("\n── Sample long queries ──\n")
    for q in long_queries[:2]:
        print(f"[{q['id']}]")
        print(f"Base: {q['base_query'][:80]}...")
        print(f"Long: {q['question'][:200]}...")
        print()


if __name__ == "__main__":
    main()
