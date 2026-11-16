# EMPIRICAL DATASET AUDIT REPORT: PHASE 9 (50,135 CLAUSES)

- **Audit Date:** 2026-11-14
- **Total Rows Audited:** 50,135
- **Document Isolation Status:** PASS (0 document overlap across all splits)

## 1. Split Distribution

| Split | Clause Count | Percentage | Unique Documents | Unique Clause Texts |
| :--- | :---: | :---: | :---: | :---: |
| **Train** | 35,105 | 70.02% | 3,633 | 21,006 |
| **Validation** | 7,500 | 14.96% | 750 | 6,152 |
| **Test** | 7,530 | 15.02% | 753 | 6,105 |
| **Total** | **50,135** | **100.00%** | **5,136** | **26,706** |

## 2. Domain Breakdown

| Domain | Total Clauses | Train | Validation | Test | Unique Docs | Classes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rental Agreement** | 16,696 | 11,686 | 2,500 | 2,510 | 1,697 | 10 |
| **Job Offer Letter** | 16,742 | 11,732 | 2,500 | 2,510 | 1,742 | 10 |
| **Insurance Policy** | 16,697 | 11,687 | 2,500 | 2,510 | 1,697 | 10 |

## 3. Cross-Split Overlap & Leakage Verification

- **Train ∩ Validation Document Overlap:** 0 documents
- **Train ∩ Test Document Overlap:** 0 documents
- **Validation ∩ Test Document Overlap:** 0 documents
- **Document Isolation Verdict:** **PASS (0 document overlap across all splits)**

- **Train ∩ Test Verbatim Text Overlap:** 3153 clauses
- **Validation ∩ Test Verbatim Text Overlap:** 1350 clauses

## 4. Class Balance Summary
Each domain contains exactly 10 canonical clause categories and 3 favorability categories (fair, needs_review, unfavorable).
