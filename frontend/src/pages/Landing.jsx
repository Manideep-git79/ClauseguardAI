import React from "react";
import {
  ShieldAlert,
  Lock,
  Cpu,
  FileCheck,
  Search,
  Clock,
  ArrowRight,
  CheckCircle,
  FileText,
  Briefcase,
  Home,
  Shield,
  Layers,
} from "lucide-react";

export function Landing({ onNavigate }) {
  const documentTypes = [
    {
      id: "rental_agreement",
      title: "Residential Lease Agreements",
      icon: Home,
      description:
        "Analyzes security deposit limits, landlord entry notice windows, repair responsibilities, auto-renewal traps, and early termination penalties.",
      clauses: ["Security Deposit", "Late Fees", "Entry Notice", "Maintenance", "Subletting"],
    },
    {
      id: "job_offer_letter",
      title: "Employment Offer Letters",
      icon: Briefcase,
      description:
        "Detects aggressive non-compete agreements, broad IP and invention assignment, at-will employment clauses, and unverified bonus contingencies.",
      clauses: ["Non-Compete", "At-Will Employment", "IP Assignment", "Bonus Contingencies", "Dispute Resolution"],
    },
    {
      id: "insurance_policy",
      title: "Insurance Policy Specimens",
      icon: Shield,
      description:
        "Audits property/casualty specimens (HO-4 broad form) for restrictive exclusions, strict loss notification deadlines, and subrogation liabilities.",
      clauses: ["Exclusions", "Loss Reporting", "Deductibles", "Cancellation", "Subrogation"],
    },
  ];

  const corePillars = [
    {
      icon: Lock,
      title: "Zero-PII Leakage Architecture",
      text: "Microsoft Presidio redacts personal identifying info (names, phone numbers, SSNs, locations) before segmentation, classification, or vector embedding. Downstream models only ever process anonymized placeholders.",
    },
    {
      icon: Cpu,
      title: "Empirical 3-Way Model Evaluation",
      text: "Tested Baseline TF-IDF+LR against BERT-no-context and BERT-windowed on authentic public documents. Evaluated on macro-F1 with Cohen's Kappa adjudication to wire the highest-performing model per task.",
    },
    {
      icon: Layers,
      title: "Config-Driven Risk Scoring",
      text: "Deterministic composite formula in risk_config.json weighing clause severity, favorability confidence, and missing checklist items. Zero black-box magic numbers.",
    },
    {
      icon: Search,
      title: "Grounded Legal RAG Engine",
      text: "PostgreSQL pgvector (768-dim embeddings) searches relevant clauses for follow-up questions. Our AI is strictly prompted to cite [Clause X] or declare ungrounded if not in the document.",
    },
    {
      icon: Clock,
      title: "Proactive Deadline Auditing",
      text: "spaCy NER and regex surface 30-day notice periods, renewal windows, and grace periods, categorizing confidence and flagging ambiguous dates for human review.",
    },
    {
      icon: FileCheck,
      title: "Executive PDF Reports",
      text: "Generates formatted PDF documentation containing executive risk scores, clause audits, PII redaction metrics, and missing clause alerts.",
    },
  ];

  return (
    <div className="space-y-16 py-4">
      <section className="relative border border-neutral-200 bg-neutral-50 p-8 sm:p-12 lg:p-16 rounded-xl shadow-sm">
        <div className="max-w-3xl">
          <div className="inline-flex items-center gap-2 px-3 py-1 text-xs font-mono border border-neutral-300 bg-white text-black uppercase tracking-wider mb-6 rounded font-semibold shadow-sm">
            <ShieldAlert className="w-3.5 h-3.5 text-black" />
            <span>Automated Legal Risk Intelligence</span>
          </div>

          <h1 className="font-serif text-4xl sm:text-5xl lg:text-6xl font-bold tracking-tight text-black leading-[1.1] mb-6">
            Uncover Hidden Liabilities in Legal Agreements
          </h1>

          <p className="text-base sm:text-lg text-neutral-600 font-sans leading-relaxed mb-8">
            ClauseGuard AI segments legal contracts, classifies clause favorability with dual-model evaluation, computes transparent risk scores, detects missing protective terms, and powers grounded follow-up inquiries.
          </p>

          <div className="flex flex-wrap items-center gap-4">
            <button
              onClick={() => onNavigate("upload")}
              className="px-6 py-3.5 text-xs font-mono uppercase tracking-wider border border-black bg-black text-white hover:bg-neutral-800 transition-colors flex items-center gap-2 font-semibold rounded shadow-sm"
            >
              <span>Upload Agreement</span>
              <ArrowRight className="w-4 h-4" />
            </button>

            <button
              onClick={() => onNavigate("documents")}
              className="px-6 py-3.5 text-xs font-mono uppercase tracking-wider border border-neutral-300 bg-white text-black hover:border-black hover:bg-neutral-100 transition-colors font-medium rounded shadow-sm"
            >
              View Analyzed Documents
            </button>
          </div>
        </div>

        <div className="mt-12 pt-6 border-t border-neutral-200 grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs font-mono text-neutral-600">
          <div>
            <span className="text-[10px] text-neutral-500 block uppercase font-semibold">Privacy</span>
            <span className="text-black font-medium">Presidio Redaction</span>
          </div>
          <div>
            <span className="text-[10px] text-neutral-500 block uppercase font-semibold">Classifier</span>
            <span className="text-black font-medium">TF-IDF + BERT</span>
          </div>
          <div>
            <span className="text-[10px] text-neutral-500 block uppercase font-semibold">Vector Store</span>
            <span className="text-black font-medium">pgvector (768-dim)</span>
          </div>
          <div>
            <span className="text-[10px] text-neutral-500 block uppercase font-semibold">RAG Engine</span>
            <span className="text-black font-medium">Deciva AI Assistant</span>
          </div>
        </div>
      </section>

      <section className="space-y-6">
        <div className="border-b border-neutral-200 pb-3">
          <span className="text-xs font-mono text-neutral-500 uppercase tracking-wider block font-semibold">
            Target Taxonomies
          </span>
          <h2 className="font-serif text-2xl font-bold text-black">
            Supported Document Formats & Checklists
          </h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {documentTypes.map((dt) => {
            const Icon = dt.icon;
            return (
              <div
                key={dt.id}
                className="border border-neutral-200 bg-white p-6 flex flex-col justify-between hover:border-black transition-colors rounded-xl shadow-sm"
              >
                <div>
                  <div className="w-10 h-10 border border-neutral-300 bg-neutral-100 flex items-center justify-center text-black mb-4 rounded-lg">
                    <Icon className="w-5 h-5 text-neutral-700" />
                  </div>
                  <h3 className="font-serif font-bold text-xl text-black mb-2">
                    {dt.title}
                  </h3>
                  <p className="text-xs text-neutral-600 leading-relaxed mb-6 font-sans">
                    {dt.description}
                  </p>
                </div>

                <div>
                  <span className="text-[10px] font-mono text-neutral-500 uppercase tracking-wider block mb-2 font-semibold">
                    Key Evaluated Clauses:
                  </span>
                  <div className="flex flex-wrap gap-1.5 mb-6">
                    {dt.clauses.map((c, i) => (
                      <span
                        key={i}
                        className="text-[11px] font-mono border border-neutral-200 bg-neutral-100 px-2 py-0.5 text-neutral-800 rounded"
                      >
                        {c}
                      </span>
                    ))}
                  </div>

                  <button
                    onClick={() => onNavigate("upload", { defaultDocType: dt.id })}
                    className="w-full py-2.5 text-xs font-mono uppercase tracking-wider border border-neutral-300 hover:border-black bg-white text-black hover:bg-neutral-50 transition-colors font-semibold rounded"
                  >
                    Analyze this type →
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      </section>

      <section className="space-y-6">
        <div className="border-b border-neutral-200 pb-3">
          <span className="text-xs font-mono text-neutral-500 uppercase tracking-wider block font-semibold">
            System Architecture
          </span>
          <h2 className="font-serif text-2xl font-bold text-black">
            Built on Rigorous Engineering Principles
          </h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {corePillars.map((p, idx) => {
            const Icon = p.icon;
            return (
              <div key={idx} className="border border-neutral-200 bg-white p-6 rounded-xl shadow-sm">
                <div className="w-8 h-8 border border-neutral-300 bg-neutral-100 flex items-center justify-center text-black mb-4 rounded-lg">
                  <Icon className="w-4 h-4 text-neutral-700" />
                </div>
                <h3 className="font-serif font-bold text-base text-black mb-2">
                  {p.title}
                </h3>
                <p className="text-xs text-neutral-600 leading-relaxed font-sans">
                  {p.text}
                </p>
              </div>
            );
          })}
        </div>
      </section>

      <section className="border border-neutral-200 bg-neutral-50 p-6 sm:p-8 rounded-xl shadow-sm">
        <div className="flex flex-wrap items-center justify-between gap-4 border-b border-neutral-200 pb-4 mb-6">
          <div>
            <span className="text-[10px] font-mono text-neutral-500 uppercase tracking-wider block font-semibold">
              Held-Out Test Set Performance
            </span>
            <h3 className="font-serif font-bold text-xl text-black">
              Dual-Model Selection Benchmark (F1-Macro)
            </h3>
          </div>
          <span className="text-xs font-mono border border-neutral-300 bg-white text-black px-2.5 py-1 rounded font-semibold shadow-sm">
            VERIFIED NO-HALLUCINATION DATASET
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 font-mono text-xs">
          <div className="border border-neutral-200 p-4 bg-white rounded-lg shadow-sm">
            <span className="text-neutral-500 block text-[10px] uppercase font-semibold">Residential Leases</span>
            <div className="text-black font-serif text-lg font-bold mt-1">F1: 0.7333 (Baseline)</div>
            <p className="text-[11px] text-neutral-600 mt-2 font-sans leading-relaxed">
              TF-IDF + LR selected for clause typing (15 classes); BERT-no-context selected for favorability scoring (+0.041 F1 lift).
            </p>
          </div>
          <div className="border border-neutral-200 p-4 bg-white rounded-lg shadow-sm">
            <span className="text-neutral-500 block text-[10px] uppercase font-semibold">Job Offer Letters</span>
            <div className="text-black font-serif text-lg font-bold mt-1">F1: 0.6667 (Baseline)</div>
            <p className="text-[11px] text-neutral-600 mt-2 font-sans leading-relaxed">
              TF-IDF + LR selected for clause typing (13 classes); baseline selected for favorability.
            </p>
          </div>
          <div className="border border-neutral-200 p-4 bg-white rounded-lg shadow-sm">
            <span className="text-neutral-500 block text-[10px] uppercase font-semibold">Insurance Specimen (HO-4)</span>
            <div className="text-black font-serif text-lg font-bold mt-1">F1: 0.4848 (Baseline)</div>
            <p className="text-[11px] text-neutral-600 mt-2 font-sans leading-relaxed">
              TF-IDF + LR for clause typing; BERT-no-context won on favorability (0.4762 vs 0.2963, +0.18 lift).
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}

export default Landing;
