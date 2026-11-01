import sys
from pathlib import Path

# Add backend directory to path
backend_dir = Path("backend")
sys.path.insert(0, str(backend_dir))

from app.services.nlp.classifier_bert import get_bert_model_and_tokenizer, classify_clause, classify_clauses_batch

real_test_cases = [
    # Rental agreement
    ("Tenant shall deposit with Landlord the sum of $2,500.00 as a Security Deposit upon signing this Agreement.", "rental_agreement", "security_deposit"),
    ("Rent shall be payable in advance on or before the first day of each calendar month in the amount of $1,850.00.", "rental_agreement", "rent_payment_terms"),
    ("Tenant shall incur an immediate late charge of $75.00, plus an additional recurring daily penalty of $15.00.", "rental_agreement", "late_fees_penalty"),
    ("Landlord or its authorized contractors may enter the leased premises upon providing reasonable advance written notice of at least 24 hours.", "rental_agreement", "entry_notice_access"),
    ("Tenant shall not assign this lease, sublet the premises, or grant any license to occupy without Landlord consent.", "rental_agreement", "subletting"),
    ("No animals, dogs, cats, or pets of any kind shall be kept on the premises without Landlord authorization.", "rental_agreement", "pet_policy"),
    ("Tenant agrees to indemnify, defend, and hold harmless Landlord from any claims or liabilities.", "rental_agreement", "indemnification"),

    # Job offer letter
    ("Your starting annual base salary will be $185,000.00, payable in semi-monthly installments.", "job_offer_letter", "compensation_salary"),
    ("Your employment with the Company is strictly at-will. Either party may terminate at any time.", "job_offer_letter", "at_will_employment"),
    ("You will be eligible to participate in the discretionary annual incentive plan with a target bonus of 20%.", "job_offer_letter", "bonus_incentive"),
    ("You agree that all inventions, software code, and trade secrets developed by you belong exclusively to Company.", "job_offer_letter", "confidentiality_nda"),
    ("During your employment and for 12 months following termination, you agree not to compete with the Company.", "job_offer_letter", "non_compete_non_solicit"),
    ("Your anticipated start date is November 16, 2026. This offer is contingent upon background check.", "job_offer_letter", "start_date_contingencies"),

    # Insurance policy
    ("We cover personal property owned or used by an insured while it is anywhere in the world.", "insurance_policy", "coverage_scope"),
    ("We do not insure for loss caused directly or indirectly by water damage, flood, or earth movement.", "insurance_policy", "exclusions"),
    ("The standard deductible is $500.00 per covered occurrence.", "insurance_policy", "deductible_premium"),
    ("In case of a loss to covered property, you must give prompt written notice to us within thirty days.", "insurance_policy", "claims_process"),
    ("If you and we fail to agree on the amount of loss, either party may demand an appraisal of the loss.", "insurance_policy", "dispute_resolution_appraisal"),
    ("An insured may waive in writing before a loss all rights of recovery against any person.", "insurance_policy", "subrogation"),
]

print("=" * 65)
print("TESTING LIVE DISTILBERT INFERENCE ON UNSEEN CONTRACT CLAUSES")
print("=" * 65)

correct = 0
total = len(real_test_cases)

for text, domain, expected in real_test_cases:
    pred_label, conf, model_ver = classify_clause(text, document_type=domain, task="clause_type")
    is_correct = (pred_label == expected)
    if is_correct:
        correct += 1
    mark = "[PASS]" if is_correct else "[FAIL]"
    print(f"{mark} [{domain}] Expected: {expected:28} | Predicted: {pred_label:28} ({conf:.2%}) | Version: {model_ver}")

acc = (correct / total) * 100
print("\n" + "=" * 65)
print(f"LIVE DISTILBERT UNSEEN REAL ACCURACY: {correct}/{total} = {acc:.2f}%")
print("=" * 65)

assert acc == 100.0, f"Expected 100% accuracy, got {acc}%"
print("\n>>> ALL DISTILBERT LIVE INFERENCE TESTS PASSED WITH 100% ACCURACY! <<<")
