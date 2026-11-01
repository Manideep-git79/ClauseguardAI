import React, { useState } from "react";
import { Badge } from "../common/Badge";
import { AlertCircle, Copy, Check, ChevronDown, ChevronUp } from "lucide-react";

export function ClauseCard({ clause, displayIndex, isSelected, onSelect }) {
  const [copied, setCopied] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);

  const copyToClipboard = (e) => {
    e.stopPropagation();
    navigator.clipboard.writeText(clause.redacted_text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const isLong = clause.redacted_text.length > 280;
  const displayText = isLong && !isExpanded
    ? clause.redacted_text.slice(0, 280) + "..."
    : clause.redacted_text;

  const riskColor =
    clause.risk_score >= 70
      ? "text-red-800 border-red-300 bg-red-50"
      : clause.risk_score >= 40
      ? "text-amber-800 border-amber-300 bg-amber-50"
      : "text-green-800 border-green-300 bg-green-50";

  const isNeedsReview = clause.human_review || clause.favorability_label === "needs_review";

  return (
    <div
      onClick={onSelect}
      className={`border p-5 transition-all cursor-pointer shadow-sm ${
        isSelected
          ? "border-black bg-neutral-100"
          : "border-neutral-200 bg-white hover:border-neutral-400"
      }`}
    >
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-neutral-200 pb-3 mb-3">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="font-mono text-xs text-neutral-500 font-semibold">
            #{displayIndex !== undefined ? displayIndex : clause.clause_index + 1}
          </span>
          <span className="font-serif font-bold text-base text-black">
            {clause.clause_type ? clause.clause_type.replace(/_/g, " ").toUpperCase() : "UNCLASSIFIED"}
          </span>
          {clause.page_number && (
            <span className="font-mono text-[11px] text-neutral-600 border border-neutral-300 px-1.5 py-0.5 bg-neutral-50 font-medium">
              Page {clause.page_number}
            </span>
          )}
          <Badge variant={clause.favorability_label || "fair"}>
            {clause.favorability_label || "fair"}
          </Badge>

          {isNeedsReview && (
            <span className="inline-flex items-center gap-1 text-[11px] font-mono text-amber-800 border border-amber-300 bg-amber-50 px-2 py-0.5 font-semibold">
              <AlertCircle className="w-3 h-3 text-amber-600" />
              HUMAN REVIEW FLAGGED
            </span>
          )}
        </div>

        <div className="flex items-center gap-3">
          <div className="text-right">
            <span className="text-[10px] font-mono text-neutral-500 block uppercase font-semibold">
              Clause Risk
            </span>
            <span className={`text-xs font-mono font-bold px-2 py-0.5 border ${riskColor}`}>
              {clause.risk_score} / 100
            </span>
          </div>

          <button
            onClick={copyToClipboard}
            className="p-1.5 border border-neutral-300 hover:border-black text-neutral-600 hover:text-black bg-white transition-colors"
            title="Copy redacted text"
          >
            {copied ? <Check className="w-3.5 h-3.5 text-green-600" /> : <Copy className="w-3.5 h-3.5" />}
          </button>
        </div>
      </div>

      {isNeedsReview && clause.filter_reason && (
        <div className="mb-3 px-2.5 py-1.5 bg-amber-50 border border-amber-200 text-[11px] font-mono text-amber-900 rounded-none flex items-start gap-1.5">
          <span className="font-bold whitespace-nowrap">Review Reason:</span>
          <span>{clause.filter_reason}</span>
        </div>
      )}

      <p className="text-xs text-neutral-800 font-sans leading-relaxed whitespace-pre-wrap">
        {displayText}
      </p>

      {isLong && (
        <button
          onClick={(e) => {
            e.stopPropagation();
            setIsExpanded(!isExpanded);
          }}
          className="mt-3 text-[11px] font-mono text-neutral-600 hover:text-black flex items-center gap-1 font-semibold"
        >
          {isExpanded ? (
            <>
              Show less <ChevronUp className="w-3 h-3" />
            </>
          ) : (
            <>
              Show full clause text <ChevronDown className="w-3 h-3" />
            </>
          )}
        </button>
      )}
    </div>
  );
}
