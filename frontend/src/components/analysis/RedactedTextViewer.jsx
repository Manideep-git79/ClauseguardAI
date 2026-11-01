import React, { useState } from "react";
import { ShieldCheck, Copy, Check } from "lucide-react";

function renderWithPseudonymBadges(text) {
  if (!text) return "";
  const regex = /(\[(?:Person|Email|Phone|Location|SSN|Credit Card|Bank Account|IP Address)\s+\d+\])/g;
  const parts = text.split(regex);
  return parts.map((part, i) => {
    if (part.match(/^\[(?:Person|Email|Phone|Location|SSN|Credit Card|Bank Account|IP Address)\s+\d+\]$/)) {
      return (
        <span
          key={i}
          className="bg-neutral-200 border border-neutral-300 text-neutral-900 font-mono font-bold px-1.5 py-0.5 rounded text-[11px] inline-block mx-0.5"
        >
          {part}
        </span>
      );
    }
    return part;
  });
}

export function RedactedTextViewer({ redactedText = "", clauses = [], selectedClauseId, onSelectClause }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(redactedText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="border border-neutral-200 bg-white p-6 space-y-4 shadow-sm">
      <div className="flex items-center justify-between border-b border-neutral-200 pb-3">
        <div className="flex items-center gap-2">
          <ShieldCheck className="w-5 h-5 text-green-600" />
          <div>
            <span className="text-[11px] font-mono uppercase tracking-widest text-neutral-500 block font-semibold">
              Redacted Document Text
            </span>
            <h3 className="font-serif text-lg font-bold text-black">
              Full Text (PII-Clean & Pseudonymized)
            </h3>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <span className="text-[11px] font-mono text-green-800 border border-green-300 bg-green-50 px-2.5 py-1 font-semibold">
            ALL PII PSEUDONYMIZED
          </span>
          <button
            onClick={handleCopy}
            className="px-3 py-1.5 border border-neutral-300 hover:border-black text-xs font-mono text-neutral-700 hover:text-black bg-white flex items-center gap-1.5 transition-colors"
          >
            {copied ? <Check className="w-3.5 h-3.5 text-green-600" /> : <Copy className="w-3.5 h-3.5" />}
            {copied ? "Copied" : "Copy Text"}
          </button>
        </div>
      </div>

      <p className="text-xs text-neutral-600 leading-relaxed font-sans">
        This is the sole version of the document text processed downstream by machine learning models and displayed in the interface.
        Raw text is encrypted at rest for audit-only purposes.
      </p>

      <div className="border border-neutral-300 bg-neutral-50 p-5 max-h-[550px] overflow-y-auto font-mono text-xs leading-relaxed space-y-4 text-black">
        {clauses.length > 0 ? (
          clauses.map((c) => {
            const isSelected = selectedClauseId === c.id;
            const isUnfavorable = c.favorability_label === "unfavorable";
            const isReview = c.favorability_label === "needs_review";

            let bgClass = "hover:bg-neutral-100";
            if (isSelected) {
              bgClass = "bg-neutral-200 border-l-2 border-black pl-2";
            } else if (isUnfavorable) {
              bgClass = "bg-red-50 border-l-2 border-red-500 pl-2";
            } else if (isReview) {
              bgClass = "bg-amber-50 border-l-2 border-amber-500 pl-2";
            }

            return (
              <div
                key={c.id}
                onClick={() => onSelectClause(c.id)}
                className={`p-2 transition-colors cursor-pointer rounded-sm ${bgClass}`}
              >
                <div className="flex items-center gap-2 mb-1 text-[10px] text-neutral-500">
                  <span className="text-neutral-700 font-bold">Clause #{c.clause_index + 1}</span>
                  <span>•</span>
                  <span className="uppercase text-neutral-600">
                    {c.clause_type ? c.clause_type.replace(/_/g, " ") : "General"}
                  </span>
                  <span>•</span>
                  <span>Risk {c.risk_score}</span>
                </div>
                <div className="text-neutral-800 whitespace-pre-wrap leading-relaxed">
                  {renderWithPseudonymBadges(c.redacted_text)}
                </div>
              </div>
            );
          })
        ) : (
          <div className="whitespace-pre-wrap text-neutral-800 leading-relaxed">
            {renderWithPseudonymBadges(redactedText) || "No text available."}
          </div>
        )}
      </div>
    </div>
  );
}
