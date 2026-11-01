import React from "react";
import {
  FileText,
  UploadCloud,
  CheckCircle2,
  XCircle,
  HelpCircle,
  ArrowRight,
  ArrowDown,
  Cpu,
  Layers,
  Database,
  Search,
  Bot,
  ShieldAlert,
  ShieldCheck,
  Split,
  EyeOff,
  Clock,
  Binary,
} from "lucide-react";

/**
 * Native, responsive SVG and HTML micro-diagrams for the architecture flowchart.
 * Pure monochrome, editorial aesthetic (neutral-900, neutral-600, border-neutral-300, bg-neutral-50).
 */
export function MiniDiagram({ type }) {
  switch (type) {
    case "upload_pipeline":
      return (
        <div className="border border-neutral-200 bg-neutral-50/70 p-4 rounded-lg my-3 text-xs font-mono">
          <div className="text-[11px] font-sans font-semibold text-neutral-700 uppercase tracking-wider mb-2">
            Payload Flow: Client to Gateway
          </div>
          <div className="flex flex-col sm:flex-row items-center justify-between gap-3 text-center">
            <div className="bg-white border border-neutral-300 p-2.5 rounded w-full sm:w-1/3 shadow-sm">
              <UploadCloud className="w-5 h-5 mx-auto mb-1 text-neutral-800" />
              <div className="font-semibold text-neutral-900">Dropzone (React)</div>
              <div className="text-[10px] text-neutral-600 mt-0.5">File &lt; 20MB, Valid Ext</div>
            </div>
            <div className="flex items-center text-neutral-600 font-bold">
              <span className="hidden sm:inline">→</span>
              <span className="sm:hidden">↓</span>
              <span className="text-[10px] px-1 font-normal">multipart/form-data</span>
              <span className="hidden sm:inline">→</span>
              <span className="sm:hidden">↓</span>
            </div>
            <div className="bg-white border border-neutral-300 p-2.5 rounded w-full sm:w-1/3 shadow-sm">
              <FileText className="w-5 h-5 mx-auto mb-1 text-neutral-800" />
              <div className="font-semibold text-neutral-900">POST /api/documents</div>
              <div className="text-[10px] text-neutral-600 mt-0.5">Flask Blueprint Handler</div>
            </div>
          </div>
        </div>
      );

    case "validation_flow":
      return (
        <div className="border border-neutral-200 bg-neutral-50/70 p-4 rounded-lg my-3 text-xs font-mono">
          <div className="text-[11px] font-sans font-semibold text-neutral-700 uppercase tracking-wider mb-2">
            Multi-Layer Ingestion Validation
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-4 gap-2 text-center">
            <div className="bg-white border border-neutral-200 p-2 rounded">
              <div className="text-[10px] text-neutral-600 uppercase">Step 1</div>
              <div className="font-semibold text-neutral-900 mt-0.5">JWT Auth</div>
              <div className="text-[10px] text-neutral-600">Bearer Token</div>
            </div>
            <div className="bg-white border border-neutral-200 p-2 rounded">
              <div className="text-[10px] text-neutral-600 uppercase">Step 2</div>
              <div className="font-semibold text-neutral-900 mt-0.5">Magic Bytes</div>
              <div className="text-[10px] text-neutral-600">%PDF, PK, etc.</div>
            </div>
            <div className="bg-white border border-neutral-200 p-2 rounded">
              <div className="text-[10px] text-neutral-600 uppercase">Step 3</div>
              <div className="font-semibold text-neutral-900 mt-0.5">SHA-256 Hash</div>
              <div className="text-[10px] text-neutral-600">Storage Integrity</div>
            </div>
            <div className="bg-white border border-neutral-200 p-2 rounded">
              <div className="text-[10px] text-neutral-600 uppercase">Step 4</div>
              <div className="font-semibold text-neutral-900 mt-0.5">DB Init</div>
              <div className="text-[10px] text-neutral-600">status="processing"</div>
            </div>
          </div>
        </div>
      );

    case "dispatcher_flow":
      return (
        <div className="border border-neutral-200 bg-neutral-50/70 p-4 rounded-lg my-3 text-xs font-mono">
          <div className="text-[11px] font-sans font-semibold text-neutral-700 uppercase tracking-wider mb-2">
            MIME & Extension Routing Matrix
          </div>
          <div className="flex flex-col sm:flex-row items-center justify-between gap-2">
            <div className="bg-white border border-neutral-300 p-2 rounded text-center w-full sm:w-1/4">
              <span className="font-bold text-neutral-900">.pdf</span>
              <div className="text-[10px] text-neutral-600">PyMuPDF + RapidOCR</div>
            </div>
            <div className="bg-white border border-neutral-300 p-2 rounded text-center w-full sm:w-1/4">
              <span className="font-bold text-neutral-900">.docx</span>
              <div className="text-[10px] text-neutral-600">python-docx + Tables</div>
            </div>
            <div className="bg-white border border-neutral-300 p-2 rounded text-center w-full sm:w-1/4">
              <span className="font-bold text-neutral-900">.txt</span>
              <div className="text-[10px] text-neutral-600">UTF-8 Normalizer</div>
            </div>
            <div className="bg-white border border-neutral-300 p-2 rounded text-center w-full sm:w-1/4">
              <span className="font-bold text-neutral-900">.png / .jpg</span>
              <div className="text-[10px] text-neutral-600">Direct RapidOCR</div>
            </div>
          </div>
        </div>
      );

    case "native_extraction":
      return (
        <div className="border border-neutral-200 bg-neutral-50/70 p-4 rounded-lg my-3 text-xs font-mono">
          <div className="text-[11px] font-sans font-semibold text-neutral-700 uppercase tracking-wider mb-2">
            Native Stream Extraction & Spatial Layout
          </div>
          <div className="flex flex-col sm:flex-row items-center justify-between gap-3 text-center">
            <div className="bg-white border border-neutral-300 p-2 rounded w-full sm:w-1/3">
              <div className="font-bold text-neutral-900">PyMuPDF page.get_text()</div>
              <div className="text-[10px] text-neutral-600">dict mode: blocks, lines, spans</div>
            </div>
            <span className="text-neutral-500 font-bold">→</span>
            <div className="bg-white border border-neutral-300 p-2 rounded w-full sm:w-1/3">
              <div className="font-bold text-neutral-900">Coordinates (x0, y0, x1, y1)</div>
              <div className="text-[10px] text-neutral-600">Sorted Top-to-Bottom, Left-to-Right</div>
            </div>
            <span className="text-neutral-500 font-bold">→</span>
            <div className="bg-white border border-neutral-300 p-2 rounded w-full sm:w-1/3">
              <div className="font-bold text-neutral-900">assess_page_extraction_quality()</div>
              <div className="text-[10px] text-neutral-600">Calculates Printable Text Ratio</div>
            </div>
          </div>
        </div>
      );

    case "decision_ocr":
      return (
        <div className="border border-neutral-300 bg-neutral-50/80 p-4 rounded-lg my-3 text-xs font-mono">
          <div className="text-[11px] font-sans font-semibold text-neutral-800 uppercase tracking-wider mb-3 text-center">
            Decision: Conditional RapidOCR Fallback Gate
          </div>
          <div className="flex flex-col items-center">
            {/* Diamond Node */}
            <div className="bg-white border-2 border-neutral-800 px-4 py-2 rounded-lg text-center shadow-sm max-w-sm">
              <div className="text-[10px] text-neutral-500 uppercase tracking-widest font-sans font-semibold">
                Decision Condition
              </div>
              <div className="font-bold text-neutral-900 text-xs mt-0.5">
                Is native page text &gt;= 80 chars & printable &gt;= 85%?
              </div>
            </div>

            {/* Branches */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 w-full mt-3">
              <div className="border border-neutral-300 bg-white p-3 rounded text-left">
                <div className="flex items-center gap-1.5 text-neutral-900 font-bold text-xs">
                  <CheckCircle2 className="w-3.5 h-3.5 text-neutral-700" />
                  <span>Branch A: YES (Usable Native Text)</span>
                </div>
                <p className="text-[11px] text-neutral-600 mt-1 font-sans">
                  Retains native PyMuPDF / docx text. RapidOCR is skipped to save CPU/GPU cycles and preserve exact typography.
                </p>
                <div className="mt-2 text-[10px] bg-neutral-100 p-1 rounded text-neutral-700">
                  ocr_performed = False | high_confidence
                </div>
              </div>

              <div className="border border-neutral-300 bg-white p-3 rounded text-left">
                <div className="flex items-center gap-1.5 text-neutral-900 font-bold text-xs">
                  <XCircle className="w-3.5 h-3.5 text-neutral-700" />
                  <span>Branch B: NO (Scanned or Low Quality)</span>
                </div>
                <p className="text-[11px] text-neutral-600 mt-1 font-sans">
                  Renders page at 200 DPI or extracts embedded images. RapidOCR + ONNX detects bounding boxes and merges text.
                </p>
                <div className="mt-2 text-[10px] bg-neutral-100 p-1 rounded text-neutral-700">
                  ocr_performed = True | rapidocr_onnx
                </div>
              </div>
            </div>
          </div>
        </div>
      );

    case "normalization_flow":
      return (
        <div className="border border-neutral-200 bg-neutral-50/70 p-4 rounded-lg my-3 text-xs font-mono">
          <div className="text-[11px] font-sans font-semibold text-neutral-700 uppercase tracking-wider mb-2">
            Reversible Normalization Pipeline
          </div>
          <div className="flex flex-col sm:flex-row items-center justify-between gap-2 text-center">
            <div className="bg-white border border-neutral-200 p-2 rounded w-full sm:w-1/3">
              <div className="font-semibold text-neutral-900">Unicode NFKC</div>
              <div className="text-[10px] text-neutral-600">Standardizes smart quotes & ligatures</div>
            </div>
            <span className="text-neutral-400">→</span>
            <div className="bg-white border border-neutral-200 p-2 rounded w-full sm:w-1/3">
              <div className="font-semibold text-neutral-900">\r\n → \n</div>
              <div className="text-[10px] text-neutral-600">Unifies CRLF line breaks</div>
            </div>
            <span className="text-neutral-400">→</span>
            <div className="bg-white border border-neutral-200 p-2 rounded w-full sm:w-1/3">
              <div className="font-semibold text-neutral-900">Offset Mapping</div>
              <div className="text-[10px] text-neutral-600">Preserves original start & end chars</div>
            </div>
          </div>
        </div>
      );

    case "split_mixed_line":
      return (
        <div className="border border-neutral-300 bg-neutral-50/80 p-4 rounded-lg my-3 text-xs font-mono">
          <div className="text-[11px] font-sans font-semibold text-neutral-800 uppercase tracking-wider mb-2 text-center">
            Verified Block Splitting: try_split_mixed_line()
          </div>
          <div className="space-y-3">
            <div className="bg-white border border-neutral-300 p-2 rounded">
              <span className="text-[10px] uppercase text-neutral-500 font-sans font-semibold">Raw Input Line:</span>
              <div className="text-neutral-900 font-semibold mt-0.5">
                "PAYMENT: The tenant shall pay rent on the first day of each month."
              </div>
            </div>

            <div className="flex items-center justify-center gap-2 text-neutral-500">
              <Split className="w-4 h-4 text-neutral-700" />
              <span className="text-[11px] font-sans">Regex: ^([A-Z\s]{3,35}|Section\s+\d+):\s*(.*)$</span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div className="bg-white border border-neutral-300 p-2.5 rounded">
                <span className="text-[10px] uppercase text-neutral-500 font-sans font-semibold">Block 1: Heading Metadata</span>
                <div className="font-bold text-neutral-900 mt-1">"PAYMENT"</div>
                <div className="text-[10px] text-neutral-600 mt-1">
                  content_type: HEADING<br />is_substantive: false (Gated Out)
                </div>
              </div>
              <div className="bg-white border border-neutral-300 p-2.5 rounded">
                <span className="text-[10px] uppercase text-neutral-500 font-sans font-semibold">Block 2: Operative Covenant</span>
                <div className="font-bold text-neutral-900 mt-1">"The tenant shall pay rent..."</div>
                <div className="text-[10px] text-neutral-600 mt-1">
                  content_type: CLAUSE<br />is_substantive: true (Eligible for ML)
                </div>
              </div>
            </div>
          </div>
        </div>
      );

    case "eligibility_gate":
      return (
        <div className="border border-neutral-300 bg-neutral-50/80 p-4 rounded-lg my-3 text-xs font-mono">
          <div className="text-[11px] font-sans font-semibold text-neutral-800 uppercase tracking-wider mb-3 text-center">
            Substantive Clause Eligibility Filter Gate
          </div>
          <div className="flex flex-col items-center">
            <div className="bg-white border-2 border-neutral-800 px-4 py-2 rounded-lg text-center shadow-sm max-w-sm">
              <div className="text-[10px] text-neutral-500 uppercase tracking-widest font-sans font-semibold">
                Decision Condition
              </div>
              <div className="font-bold text-neutral-900 text-xs mt-0.5">
                Does block establish binding legal duties, liabilities, or rights?
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 w-full mt-3">
              <div className="border border-neutral-300 bg-white p-3 rounded text-left">
                <div className="flex items-center gap-1.5 text-neutral-900 font-bold text-xs">
                  <EyeOff className="w-3.5 h-3.5 text-neutral-600" />
                  <span>Branch A: Excluded Non-Substantive</span>
                </div>
                <ul className="text-[10px] text-neutral-600 list-disc list-inside mt-1 space-y-0.5 font-sans">
                  <li>Document Titles & Headings</li>
                  <li>Salutations & Greeting lines</li>
                  <li>Signatures & Witness blocks</li>
                  <li>Addresses, Dates, Page numbers</li>
                </ul>
                <div className="mt-2 text-[10px] bg-neutral-100 p-1 rounded text-neutral-800 font-mono">
                  is_substantive = False | risk_score = 0.0
                </div>
              </div>

              <div className="border border-neutral-300 bg-white p-3 rounded text-left">
                <div className="flex items-center gap-1.5 text-neutral-900 font-bold text-xs">
                  <ShieldCheck className="w-3.5 h-3.5 text-neutral-700" />
                  <span>Branch B: Eligible Substantive Clause</span>
                </div>
                <ul className="text-[10px] text-neutral-600 list-disc list-inside mt-1 space-y-0.5 font-sans">
                  <li>Shall, agree, covenant, indemnify</li>
                  <li>Liability limits & liquidated damages</li>
                  <li>Termination & dispute terms</li>
                  <li>Warranties & covenants</li>
                </ul>
                <div className="mt-2 text-[10px] bg-neutral-100 p-1 rounded text-neutral-800 font-mono">
                  is_substantive = True | Forward to DistilBERT
                </div>
              </div>
            </div>
          </div>
        </div>
      );

    case "context_window":
      return (
        <div className="border border-neutral-200 bg-neutral-50/70 p-4 rounded-lg my-3 text-xs font-mono">
          <div className="text-[11px] font-sans font-semibold text-neutral-700 uppercase tracking-wider mb-2">
            3-Clause Sliding Context Window
          </div>
          <div className="flex flex-col sm:flex-row items-center justify-center gap-2 text-center">
            <div className="bg-white border border-neutral-200 p-2 rounded w-full sm:w-1/4 opacity-60">
              <span className="text-[10px] text-neutral-500 uppercase">Clause [i-1]</span>
              <div className="text-neutral-700 text-[11px] truncate">Preceding context</div>
            </div>
            <span className="text-neutral-400">+</span>
            <div className="bg-white border-2 border-neutral-800 p-2.5 rounded w-full sm:w-1/2 shadow-sm">
              <span className="text-[10px] text-neutral-800 font-bold uppercase">Target Clause [i]</span>
              <div className="text-neutral-900 font-bold text-xs truncate">Operative provision to classify</div>
            </div>
            <span className="text-neutral-400">+</span>
            <div className="bg-white border border-neutral-200 p-2 rounded w-full sm:w-1/4 opacity-60">
              <span className="text-[10px] text-neutral-500 uppercase">Clause [i+1]</span>
              <div className="text-neutral-700 text-[11px] truncate">Succeeding context</div>
            </div>
          </div>
        </div>
      );

    case "distilbert_dual_head":
      return (
        <div className="border border-neutral-300 bg-neutral-50/80 p-4 rounded-lg my-3 text-xs font-mono">
          <div className="text-[11px] font-sans font-semibold text-neutral-800 uppercase tracking-wider mb-3 text-center">
            DistilBERT Multi-Task Dual-Head Neural Architecture
          </div>
          <div className="flex flex-col items-center gap-2">
            <div className="bg-white border border-neutral-300 px-3 py-1.5 rounded text-center w-full max-w-xs shadow-sm">
              <div className="font-bold text-neutral-900">DistilBERT Tokenizer</div>
              <div className="text-[10px] text-neutral-600">Max 512 tokens [CLS] ... [SEP]</div>
            </div>

            <ArrowDown className="w-4 h-4 text-neutral-600" />

            <div className="bg-white border-2 border-neutral-800 px-4 py-2 rounded text-center w-full max-w-sm shadow-sm">
              <div className="font-bold text-neutral-900 text-xs">DistilBertModel (6 Transformer Layers)</div>
              <div className="text-[10px] text-neutral-600">768-dim pooled [CLS] contextual representation</div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full mt-2">
              <div className="bg-white border border-neutral-300 p-2.5 rounded text-left">
                <div className="font-bold text-neutral-900 text-xs">Head A: Category Classifier</div>
                <div className="text-[10px] text-neutral-600 mt-1">Linear(768, 10) + Softmax</div>
                <div className="text-[10px] text-neutral-800 font-semibold mt-1">
                  10 Classes (e.g. TERMINATION, INDEMNIFICATION)
                </div>
              </div>
              <div className="bg-white border border-neutral-300 p-2.5 rounded text-left">
                <div className="font-bold text-neutral-900 text-xs">Head B: Favorability Classifier</div>
                <div className="text-[10px] text-neutral-600 mt-1">Linear(768, 3) + Softmax</div>
                <div className="text-[10px] text-neutral-800 font-semibold mt-1">
                  Favorable, Neutral, Unfavorable
                </div>
              </div>
            </div>
          </div>
        </div>
      );

    case "risk_formula":
      return (
        <div className="border border-neutral-200 bg-neutral-50/70 p-4 rounded-lg my-3 text-xs font-mono">
          <div className="text-[11px] font-sans font-semibold text-neutral-700 uppercase tracking-wider mb-2">
            Consensus Scoring Engine: Rules vs. DistilBERT
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 text-center">
            <div className="bg-white border border-neutral-300 p-2 rounded">
              <div className="text-[10px] text-neutral-600">ML Confidence</div>
              <div className="font-bold text-neutral-900 text-sm mt-0.5">&gt;= 0.45</div>
              <div className="text-[10px] text-neutral-600">Valid prediction accepted</div>
            </div>
            <div className="bg-white border border-neutral-300 p-2 rounded">
              <div className="text-[10px] text-neutral-600">ML Confidence</div>
              <div className="font-bold text-neutral-900 text-sm mt-0.5">&lt; 0.45</div>
              <div className="text-[10px] text-neutral-600">Flagged UNCERTAIN, triggers review</div>
            </div>
            <div className="bg-white border border-neutral-300 p-2 rounded">
              <div className="text-[10px] text-neutral-600">Rule Escalation</div>
              <div className="font-bold text-neutral-900 text-sm mt-0.5">Overrides</div>
              <div className="text-[10px] text-neutral-600">Unfavorable rules force high risk</div>
            </div>
          </div>
        </div>
      );

    case "safety_gate_calc":
      return (
        <div className="border border-neutral-300 bg-neutral-50/80 p-4 rounded-lg my-3 text-xs font-mono">
          <div className="text-[11px] font-sans font-semibold text-neutral-800 uppercase tracking-wider mb-2 text-center">
            Verified Clause Risk Calculation Formula
          </div>
          <div className="bg-white border border-neutral-300 p-3 rounded font-mono text-[11px] text-neutral-900 space-y-1">
            <div className="font-bold text-neutral-800">compute_clause_risk(favorability, severity, is_substantive):</div>
            <div className="pl-3 text-neutral-600">if not is_substantive: return 0.0  # SAFETY GATE</div>
            <div className="pl-3 text-neutral-600">base = &#123; unfavorable: 0.8, neutral: 0.3, favorable: 0.1 &#125;[favorability]</div>
            <div className="pl-3 text-neutral-600">multiplier = &#123; high: 1.25, medium: 1.0, low: 0.75 &#125;[severity]</div>
            <div className="pl-3 font-semibold text-neutral-900">return min(1.0, base * multiplier)</div>
          </div>
        </div>
      );

    case "risk_aggregation":
      return (
        <div className="border border-neutral-300 bg-neutral-50/80 p-4 rounded-lg my-3 text-xs font-mono">
          <div className="text-[11px] font-sans font-semibold text-neutral-800 uppercase tracking-wider mb-2 text-center">
            Document Risk Aggregation & Band Mapping
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div className="bg-white border border-neutral-200 p-3 rounded">
              <div className="text-[10px] text-neutral-600 uppercase font-sans font-semibold">Aggregation Formula</div>
              <div className="text-[11px] text-neutral-900 font-semibold mt-1">
                Score = 0.85 × Avg(Substantive Clause Risks) + 0.15 × Missing Clause Penalty
              </div>
              <div className="text-[10px] text-neutral-600 mt-2 font-sans">
                Scaled 0 to 100. Non-substantive structural blocks (titles, salutations) are strictly omitted.
              </div>
            </div>
            <div className="bg-white border border-neutral-200 p-3 rounded">
              <div className="text-[10px] text-neutral-600 uppercase font-sans font-semibold">Calibrated Risk Bands</div>
              <div className="grid grid-cols-2 gap-1 text-[10px] mt-1">
                <span className="p-1 bg-neutral-100 rounded">0–25: Low Risk</span>
                <span className="p-1 bg-neutral-100 rounded">26–50: Medium Risk</span>
                <span className="p-1 bg-neutral-100 rounded">51–75: High Risk</span>
                <span className="p-1 bg-neutral-100 rounded">76–100: Critical Risk</span>
              </div>
              <div className="text-[10px] text-neutral-700 font-semibold mt-1.5">
                Special Band: "insufficient_content" if count == 0
              </div>
            </div>
          </div>
        </div>
      );

    case "persistence_flow":
      return (
        <div className="border border-neutral-200 bg-neutral-50/70 p-4 rounded-lg my-3 text-xs font-mono">
          <div className="text-[11px] font-sans font-semibold text-neutral-700 uppercase tracking-wider mb-2">
            Atomic PostgreSQL Persistence Transaction
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-center">
            <div className="bg-white border border-neutral-200 p-2 rounded">
              <div className="font-bold text-neutral-900">Document</div>
              <div className="text-[10px] text-neutral-600">status="completed"</div>
            </div>
            <div className="bg-white border border-neutral-200 p-2 rounded">
              <div className="font-bold text-neutral-900">Clauses</div>
              <div className="text-[10px] text-neutral-600">N clause records</div>
            </div>
            <div className="bg-white border border-neutral-200 p-2 rounded">
              <div className="font-bold text-neutral-900">Deadlines</div>
              <div className="text-[10px] text-neutral-600">Notice & penalties</div>
            </div>
            <div className="bg-white border border-neutral-200 p-2 rounded">
              <div className="font-bold text-neutral-900">AuditLog</div>
              <div className="text-[10px] text-neutral-600">DOC_ANALYZED</div>
            </div>
          </div>
        </div>
      );

    case "dashboard_layout":
      return (
        <div className="border border-neutral-200 bg-neutral-50/70 p-4 rounded-lg my-3 text-xs font-mono">
          <div className="text-[11px] font-sans font-semibold text-neutral-700 uppercase tracking-wider mb-2">
            Composite UI Component Architecture
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 text-center">
            <div className="bg-white border border-neutral-300 p-2.5 rounded">
              <div className="font-bold text-neutral-900">RiskScoreGauge.jsx</div>
              <div className="text-[10px] text-neutral-600 mt-0.5">Composite 0-100 meter with safety badge</div>
            </div>
            <div className="bg-white border border-neutral-300 p-2.5 rounded">
              <div className="font-bold text-neutral-900">ClauseCard.jsx</div>
              <div className="text-[10px] text-neutral-600 mt-0.5">Filterable clause cards with favorability pill</div>
            </div>
            <div className="bg-white border border-neutral-300 p-2.5 rounded">
              <div className="font-bold text-neutral-900">ChatTab.jsx</div>
              <div className="text-[10px] text-neutral-600 mt-0.5">Interactive grounded copilot dialog</div>
            </div>
          </div>
        </div>
      );

    default:
      return null;
  }
}
