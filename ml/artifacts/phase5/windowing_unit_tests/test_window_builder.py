"""
Unit Tests for Phase 5 Target-Preserving Window Builder.
Deciva AI / ClauseGuard AI Benchmarking Suite.

Covers the 10 mandatory edge cases specified in Section 14:
- Case 1: Very short target.
- Case 2: Very long target.
- Case 3: No previous clause.
- Case 4: No next clause.
- Case 5: First clause in document.
- Case 6: Last clause in document.
- Case 7: Target exactly fills token budget.
- Case 8: Target exceeds token budget.
- Case 9: Multiple previous/next clauses.
- Case 10: Target containing 'not', 'unless', 'except', 'shall'.
"""

import sys
from pathlib import Path
from transformers import AutoTokenizer

project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from ml.artifacts.phase5.window_builder import (
    build_target_centered_window,
    ensure_special_tokens
)

def run_all_tests():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    ensure_special_tokens(tokenizer)
    
    print("=" * 60)
    print("RUNNING PHASE 5 WINDOW BUILDER UNIT TESTS")
    print("=" * 60)
    
    passed = 0
    total = 10

    # -------------------------------------------------------------
    # Case 1: Very short target
    # -------------------------------------------------------------
    target = "Rent is due on the 1st."
    prev = "This is a residential lease agreement between Landlord and Tenant."
    nxt = "Late fees of $50 apply after the 5th day of the month."
    
    input_ids, mask, (s, e), meta = build_target_centered_window(
        target_clause=target,
        previous_clauses=prev,
        next_clauses=nxt,
        tokenizer=tokenizer,
        max_length=96,
        allocation_strategy="target_priority"
    )
    assert meta["target_fully_retained"] is True, "Case 1 failed: Target not fully retained"
    assert meta["previous_token_count"] > 0, "Case 1 failed: Prev context missing"
    assert meta["next_token_count"] > 0, "Case 1 failed: Next context missing"
    assert len(input_ids) == 96, "Case 1 failed: Length is not 96"
    print("  [PASS] Case 1: Very short target")
    passed += 1

    # -------------------------------------------------------------
    # Case 2: Very long target
    # -------------------------------------------------------------
    long_target = " ".join([
        "The Tenant shall not sublet, assign, or transfer any part of the leased premises",
        "without the prior written consent of the Landlord, which consent may be withheld",
        "for any reason or no reason at all, and any attempted assignment shall be null and void",
        "and shall constitute an incurable breach of this lease leading to immediate termination",
        "and forfeiture of the security deposit in its entirety as liquidated damages.",
        "Furthermore, Tenant covenants to maintain insurance throughout the entire term.",
        "Such insurance policy must name Landlord as an additional insured party."
    ])
    input_ids, mask, (s, e), meta = build_target_centered_window(
        target_clause=long_target,
        previous_clauses=prev,
        next_clauses=nxt,
        tokenizer=tokenizer,
        max_length=128,
        allocation_strategy="target_priority"
    )
    # Target fits in 128 tokens
    assert meta["target_fully_retained"] is True, "Case 2 failed: Target should fit in 128"
    assert meta["target_retained_token_length"] == meta["target_token_length"]
    print("  [PASS] Case 2: Very long target")
    passed += 1

    # -------------------------------------------------------------
    # Case 3: No previous clause (previous_clauses = None or "")
    # -------------------------------------------------------------
    input_ids, mask, (s, e), meta = build_target_centered_window(
        target_clause=target,
        previous_clauses=None,
        next_clauses=nxt,
        tokenizer=tokenizer,
        max_length=96,
        allocation_strategy="target_priority"
    )
    assert meta["previous_token_count"] == 0, "Case 3 failed: Prev token count should be 0"
    assert meta["next_token_count"] > 0, "Case 3 failed: Next should take available budget"
    assert meta["target_fully_retained"] is True, "Case 3 failed: Target not retained"
    print("  [PASS] Case 3: No previous clause")
    passed += 1

    # -------------------------------------------------------------
    # Case 4: No next clause (next_clauses = None or "")
    # -------------------------------------------------------------
    input_ids, mask, (s, e), meta = build_target_centered_window(
        target_clause=target,
        previous_clauses=prev,
        next_clauses=None,
        tokenizer=tokenizer,
        max_length=96,
        allocation_strategy="target_priority"
    )
    assert meta["next_token_count"] == 0, "Case 4 failed: Next token count should be 0"
    assert meta["previous_token_count"] > 0, "Case 4 failed: Prev should take available budget"
    assert meta["target_fully_retained"] is True, "Case 4 failed: Target not retained"
    print("  [PASS] Case 4: No next clause")
    passed += 1

    # -------------------------------------------------------------
    # Case 5: First clause in document
    # -------------------------------------------------------------
    first_clause = "THIS RESIDENTIAL LEASE AGREEMENT is made on this 1st day of January 2024."
    input_ids, mask, (s, e), meta = build_target_centered_window(
        target_clause=first_clause,
        previous_clauses="",
        next_clauses=["Clause 2: Rent payment terms.", "Clause 3: Security deposit."],
        tokenizer=tokenizer,
        max_length=96,
        allocation_strategy="target_priority"
    )
    assert meta["previous_token_count"] == 0, "Case 5 failed: Prev token count should be 0"
    assert meta["next_token_count"] > 0, "Case 5 failed: Next context should be included"
    assert meta["target_fully_retained"] is True, "Case 5 failed: First clause target not retained"
    print("  [PASS] Case 5: First clause in document")
    passed += 1

    # -------------------------------------------------------------
    # Case 6: Last clause in document
    # -------------------------------------------------------------
    last_clause = "IN WITNESS WHEREOF, the parties hereto have executed this Agreement as of the date first above written."
    input_ids, mask, (s, e), meta = build_target_centered_window(
        target_clause=last_clause,
        previous_clauses=["Clause 20: Governing law.", "Clause 21: Severability."],
        next_clauses=None,
        tokenizer=tokenizer,
        max_length=96,
        allocation_strategy="target_priority"
    )
    assert meta["next_token_count"] == 0, "Case 6 failed: Next token count should be 0"
    assert meta["previous_token_count"] > 0, "Case 6 failed: Prev context should be included"
    assert meta["target_fully_retained"] is True, "Case 6 failed: Last clause target not retained"
    print("  [PASS] Case 6: Last clause in document")
    passed += 1

    # -------------------------------------------------------------
    # Case 7: Target exactly fills token budget
    # -------------------------------------------------------------
    # For max_length = 96 and marker_style = "target_tag", overhead is 4 ([CLS], [TARGET], [/TARGET], [SEP]). Available = 92.
    exact_words = ["clause"] * 92
    exact_text = " ".join(exact_words)
    toks = tokenizer.encode(exact_text, add_special_tokens=False)
    exact_text_adjusted = tokenizer.decode(toks[:92])
    
    input_ids, mask, (s, e), meta = build_target_centered_window(
        target_clause=exact_text_adjusted,
        previous_clauses=prev,
        next_clauses=nxt,
        tokenizer=tokenizer,
        max_length=96,
        allocation_strategy="target_priority",
        marker_style="target_tag"
    )
    assert meta["target_fully_retained"] is True, "Case 7 failed: Exact fit target should be fully retained"
    assert meta["previous_token_count"] == 0, "Case 7 failed: No room for prev"
    assert meta["next_token_count"] == 0, "Case 7 failed: No room for next"
    assert len(input_ids) == 96, "Case 7 failed: Length should be 96"
    print("  [PASS] Case 7: Target exactly fills token budget")
    passed += 1

    # -------------------------------------------------------------
    # Case 8: Target exceeds token budget
    # -------------------------------------------------------------
    huge_words = ["obligation"] * 150
    huge_text = " ".join(huge_words)
    input_ids, mask, (s, e), meta = build_target_centered_window(
        target_clause=huge_text,
        previous_clauses=prev,
        next_clauses=nxt,
        tokenizer=tokenizer,
        max_length=96,
        allocation_strategy="target_priority"
    )
    assert meta["target_fully_retained"] is False, "Case 8 failed: Huge target cannot be fully retained"
    assert meta["target_segmented"] is True, "Case 8 failed: Segmentation flag should be True"
    assert meta["previous_token_count"] == 0, "Case 8 failed: No context allowed when target exceeds budget"
    assert meta["next_token_count"] == 0, "Case 8 failed: No context allowed when target exceeds budget"
    assert len(input_ids) == 96, "Case 8 failed: Sequence length should be 96"
    print("  [PASS] Case 8: Target exceeds token budget (segmented deterministically)")
    passed += 1

    # -------------------------------------------------------------
    # Case 9: Multiple previous/next clauses
    # -------------------------------------------------------------
    prev_list = [
        "Clause 1: Parties to agreement.",
        "Clause 2: Property description.",
        "Clause 3: Term of lease."
    ]
    next_list = [
        "Clause 5: Utilities.",
        "Clause 6: Maintenance and repairs.",
        "Clause 7: Quiet enjoyment."
    ]
    input_ids, mask, (s, e), meta = build_target_centered_window(
        target_clause=target,
        previous_clauses=prev_list,
        next_clauses=next_list,
        tokenizer=tokenizer,
        max_length=128,
        allocation_strategy="target_priority"
    )
    assert meta["target_fully_retained"] is True, "Case 9 failed: Target not fully retained"
    assert meta["previous_token_count"] > 0, "Case 9 failed: Multi-prev context missing"
    assert meta["next_token_count"] > 0, "Case 9 failed: Multi-next context missing"
    print("  [PASS] Case 9: Multiple previous/next clauses accepted and formatted")
    passed += 1

    # -------------------------------------------------------------
    # Case 10: Target containing 'not', 'unless', 'except', 'shall'
    # -------------------------------------------------------------
    neg_target = "Tenant shall NOT terminate this lease UNLESS written notice is provided EXCEPT in case of fire."
    input_ids, mask, (s, e), meta = build_target_centered_window(
        target_clause=neg_target,
        previous_clauses=prev,
        next_clauses=nxt,
        tokenizer=tokenizer,
        max_length=96,
        allocation_strategy="target_priority"
    )
    target_ids_in_window = input_ids[s : e + 1]
    decoded_target = tokenizer.decode(target_ids_in_window)
    for kw in ["not", "unless", "except", "shall"]:
        assert kw in decoded_target.lower(), f"Case 10 failed: Sensitive legal keyword '{kw}' was lost!"
    print("  [PASS] Case 10: Sensitive legal keywords ('not', 'unless', 'except', 'shall') 100% preserved")
    passed += 1

    print("=" * 60)
    print(f"ALL {passed}/{total} UNIT TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_all_tests()
