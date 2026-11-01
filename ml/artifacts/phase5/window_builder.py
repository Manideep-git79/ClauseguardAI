"""
Window Builder for Phase 5 Target-Preserving Windowed DistilBERT.
Deciva AI / ClauseGuard AI Benchmarking Suite.

Guarantees 100% retention of standard-length target clauses by allocating
context budget outward from the target rather than blind left-to-right character truncation.
"""

from typing import List, Dict, Tuple, Optional, Any, Union
from transformers import PreTrainedTokenizerBase


TARGET_TAGS = ["[TARGET]", "[/TARGET]"]
SEGMENT_TAGS = ["[TARGET]", "[PREV]", "[NEXT]"]


def ensure_special_tokens(tokenizer: PreTrainedTokenizerBase, marker_style: str = "target_tag") -> Dict[str, int]:
    """
    Ensures required special tokens are registered in the tokenizer.
    Returns a dictionary mapping marker string to its token ID.
    """
    special_dict = {}
    tokens_to_ensure = TARGET_TAGS if marker_style == "target_tag" else (SEGMENT_TAGS if marker_style == "segment_tags" else [])
    missing = [tok for tok in tokens_to_ensure if tok not in tokenizer.all_special_tokens]
    if missing:
        tokenizer.add_special_tokens({"additional_special_tokens": missing})
    
    for tok in tokens_to_ensure:
        special_dict[tok] = tokenizer.convert_tokens_to_ids(tok)
    return special_dict


def build_target_centered_window(
    target_clause: str,
    previous_clauses: Optional[Union[str, List[str]]] = None,
    next_clauses: Optional[Union[str, List[str]]] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    max_length: int = 256,
    allocation_strategy: str = "target_priority",
    marker_style: str = "target_tag",
    pad_to_max_length: bool = True,
) -> Tuple[List[int], List[int], Tuple[int, int], Dict[str, Any]]:
    """
    Constructs a target-preserving sequence window with surrounding context.

    Parameters:
    -----------
    target_clause: The central legal clause to be classified.
    previous_clauses: Preceding text or list of preceding clause strings.
    next_clauses: Subsequent text or list of subsequent clause strings.
    tokenizer: Hugging Face PreTrainedTokenizer (e.g. distilbert-base-uncased).
    max_length: Total token budget (e.g. 96, 128, 192, 256, 384, 512).
    allocation_strategy:
        - "target_only": Context budget is 0; only target clause is emitted.
        - "symmetric": Leftover context capacity is split equally between prev and next.
        - "target_priority": Target is fully allocated; leftover budget dynamically balances
                             between prev and next to maximize total context retention.
        - "immediate_clause": Respects clause boundaries; complete clauses prioritized.
    marker_style:
        - "target_tag": Uses [TARGET] and [/TARGET] around target clause (matches existing checkpoints).
        - "segment_tags": Uses [PREV], [TARGET], [NEXT] prefixes.
        - "none": Uses standard tokenizer boundaries without special marker tokens.
    pad_to_max_length: If True, pads sequence with pad_token_id up to max_length.

    Returns:
    --------
    input_ids: List of token IDs of length max_length (or actual length if not padded).
    attention_mask: List of 1s (content) and 0s (padding).
    target_token_positions: (start_idx, end_idx) inclusive token indices for target clause.
    metadata: Dictionary containing diagnostic metrics and retention status.
    """
    if tokenizer is None:
        raise ValueError("A valid tokenizer must be provided.")

    marker_map = ensure_special_tokens(tokenizer, marker_style=marker_style)
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # 1. Clean and normalize inputs
    t_text = (target_clause or "").strip()
    
    if isinstance(previous_clauses, list):
        p_text = " ".join([c.strip() for c in previous_clauses if c and c.strip()])
    else:
        p_text = (previous_clauses or "").strip()

    if isinstance(next_clauses, list):
        n_text = " ".join([c.strip() for c in next_clauses if c and c.strip()])
    else:
        n_text = (next_clauses or "").strip()

    # 2. Tokenize target clause first
    target_tokens = tokenizer.encode(t_text, add_special_tokens=False)
    target_orig_len = len(t_text)
    target_tok_len = len(target_tokens)

    # 3. Calculate target overhead
    if marker_style == "target_tag":
        target_overhead = 2  # [TARGET] and [/TARGET]
    elif marker_style == "segment_tags":
        target_overhead = 1  # [TARGET]
    else:
        target_overhead = 0
    base_overhead = 2 + target_overhead  # [CLS] and [SEP] + target markers
    avail_target_capacity = max(1, max_length - base_overhead)

    # 4. Target retention policy
    if target_tok_len <= avail_target_capacity:
        retained_target = list(target_tokens)
        target_fully_retained = True
        target_segmented = False
        target_truncated = False
    else:
        retained_target = target_tokens[:avail_target_capacity]
        target_fully_retained = False
        target_segmented = True
        target_truncated = True

    # 5. Determine leftover capacity for context
    committed_tokens = 2 + target_overhead + len(retained_target)
    remaining_budget = max(0, max_length - committed_tokens)

    prev_tokens: List[int] = []
    next_tokens: List[int] = []

    if remaining_budget > 0 and allocation_strategy != "target_only":
        raw_prev_tokens = tokenizer.encode(p_text, add_special_tokens=False) if p_text else []
        raw_next_tokens = tokenizer.encode(n_text, add_special_tokens=False) if n_text else []

        has_prev = len(raw_prev_tokens) > 0
        has_next = len(raw_next_tokens) > 0

        # Markers overhead for context
        if marker_style == "segment_tags":
            prev_marker_cost = 1 if has_prev else 0
            next_marker_cost = 1 if has_next else 0
        else:
            prev_marker_cost = 0
            next_marker_cost = 0
            
        context_capacity = max(0, remaining_budget - prev_marker_cost - next_marker_cost)

        if allocation_strategy == "symmetric":
            half_budget = context_capacity // 2
            other_budget = context_capacity - half_budget
            
            p_alloc = min(len(raw_prev_tokens), half_budget) if has_prev else 0
            n_alloc = min(len(raw_next_tokens), other_budget) if has_next else 0

        elif allocation_strategy == "target_priority":
            if not has_prev:
                p_alloc = 0
                n_alloc = min(len(raw_next_tokens), context_capacity)
            elif not has_next:
                n_alloc = 0
                p_alloc = min(len(raw_prev_tokens), context_capacity)
            else:
                half = context_capacity // 2
                rem = context_capacity - half
                if len(raw_prev_tokens) <= half:
                    p_alloc = len(raw_prev_tokens)
                    n_alloc = min(len(raw_next_tokens), context_capacity - p_alloc)
                elif len(raw_next_tokens) <= rem:
                    n_alloc = len(raw_next_tokens)
                    p_alloc = min(len(raw_prev_tokens), context_capacity - n_alloc)
                else:
                    p_alloc = half
                    n_alloc = rem

        elif allocation_strategy == "immediate_clause":
            if len(raw_prev_tokens) + len(raw_next_tokens) <= context_capacity:
                p_alloc = len(raw_prev_tokens)
                n_alloc = len(raw_next_tokens)
            else:
                half = context_capacity // 2
                p_alloc = min(len(raw_prev_tokens), half)
                n_alloc = min(len(raw_next_tokens), context_capacity - p_alloc)
        else:
            raise ValueError(f"Unknown allocation strategy: {allocation_strategy}")

        if p_alloc > 0:
            prev_tokens = raw_prev_tokens[-p_alloc:]
        if n_alloc > 0:
            next_tokens = raw_next_tokens[:n_alloc]

    # 6. Sequence Assembly
    input_ids: List[int] = [cls_id]

    if len(prev_tokens) > 0:
        if marker_style == "segment_tags":
            input_ids.append(marker_map["[PREV]"])
        input_ids.extend(prev_tokens)

    if marker_style in ["target_tag", "segment_tags"]:
        input_ids.append(marker_map["[TARGET]"])
        
    start_target_pos = len(input_ids)
    input_ids.extend(retained_target)
    end_target_pos = len(input_ids) - 1

    if marker_style == "target_tag":
        input_ids.append(marker_map["[/TARGET]"])

    if len(next_tokens) > 0:
        if marker_style == "segment_tags":
            input_ids.append(marker_map["[NEXT]"])
        input_ids.extend(next_tokens)

    input_ids.append(sep_id)

    total_non_pad_tokens = len(input_ids)

    # 7. Attention Mask & Padding
    attention_mask = [1] * total_non_pad_tokens
    if pad_to_max_length:
        pad_len = max(0, max_length - total_non_pad_tokens)
        input_ids.extend([pad_id] * pad_len)
        attention_mask.extend([0] * pad_len)

    # 8. Diagnostics Metadata
    metadata = {
        "target_original_length": target_orig_len,
        "target_token_length": target_tok_len,
        "target_retained_token_length": len(retained_target),
        "previous_token_count": len(prev_tokens),
        "next_token_count": len(next_tokens),
        "total_token_count": total_non_pad_tokens,
        "target_fully_retained": target_fully_retained,
        "target_segmented": target_segmented,
        "truncated": target_truncated,
        "strategy": allocation_strategy,
        "max_length": max_length,
        "marker_style": marker_style,
    }

    return input_ids, attention_mask, (start_target_pos, end_target_pos), metadata

