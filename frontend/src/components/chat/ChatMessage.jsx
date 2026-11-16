import React, { useState } from "react";
import { User, Bot, CheckCircle2, AlertTriangle, FileText, ChevronDown, ChevronUp, Copy, Check } from "lucide-react";
import { LegalMarkdown } from "./LegalMarkdown";

export function ChatMessage({ message, onSelectClause, clauseLookup = {} }) {
  const isUser = message.role === "user";
  const [showRetrieved, setShowRetrieved] = useState(false);
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    if (!message.content) return;
    navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div
      className={`border p-5 transition-all rounded-lg ${
        isUser
          ? "border-neutral-300 bg-neutral-100 text-black ml-4 sm:ml-12"
          : "border-neutral-200 bg-white text-black shadow-sm mr-4 sm:mr-12"
      }`}
    >
      <div className="flex items-center justify-between gap-3 border-b border-neutral-200 pb-2.5 mb-3 flex-wrap">
        <div className="flex items-center gap-2">
          {isUser ? (
            <div className="w-6 h-6 border border-neutral-400 bg-white flex items-center justify-center text-black rounded">
              <User className="w-3.5 h-3.5" />
            </div>
          ) : (
            <div className="w-6 h-6 border border-neutral-300 bg-neutral-100 flex items-center justify-center text-black rounded">
              <Bot className="w-3.5 h-3.5" />
            </div>
          )}
          <span className="font-mono text-xs font-semibold text-neutral-800 uppercase tracking-wider">
            {isUser ? "You" : "ClauseGuard AI (Grounded Legal Copilot)"}
          </span>
        </div>

        {!isUser && (
          <div className="flex items-center gap-2">
            <button
              onClick={handleCopy}
              className="inline-flex items-center gap-1 text-[11px] font-mono text-neutral-600 hover:text-black border border-neutral-300 hover:border-black bg-white px-2 py-0.5 rounded transition-colors"
              title="Copy response to clipboard"
            >
              {copied ? (
                <>
                  <Check className="w-3 h-3 text-emerald-600" />
                  <span className="text-emerald-700 font-semibold">Copied</span>
                </>
              ) : (
                <>
                  <Copy className="w-3 h-3 text-neutral-500" />
                  <span>Copy</span>
                </>
              )}
            </button>

            {message.grounded ? (
              <span className="inline-flex items-center gap-1 text-[10px] font-mono uppercase tracking-wider text-black border border-neutral-300 bg-neutral-100 px-2 py-0.5 rounded">
                <CheckCircle2 className="w-3 h-3 text-emerald-600" />
                Grounded ({message.cited_clause_ids?.length || 0} citations)
              </span>
            ) : (
              <span className="inline-flex items-center gap-1 text-[10px] font-mono uppercase tracking-wider text-neutral-700 border border-neutral-300 bg-neutral-100 px-2 py-0.5 rounded">
                <AlertTriangle className="w-3 h-3 text-amber-600" />
                General Legal Query
              </span>
            )}
          </div>
        )}
      </div>

      {isUser ? (
        <div className="text-sm sm:text-base leading-relaxed text-black font-sans whitespace-pre-wrap">
          {message.content}
        </div>
      ) : (
        <LegalMarkdown content={message.content} />
      )}

      {!isUser && message.cited_clause_ids && message.cited_clause_ids.length > 0 && (
        <div className="mt-4 pt-3 border-t border-neutral-200">
          <span className="text-[10px] font-mono uppercase tracking-wider text-neutral-600 block mb-2 font-semibold">
            Directly Cited Clauses in Document:
          </span>
          <div className="flex flex-wrap gap-2">
            {message.cited_clause_ids.map((clauseId) => {
              const clause = clauseLookup[clauseId];
              return (
                <button
                  key={clauseId}
                  onClick={() => onSelectClause && onSelectClause(clauseId)}
                  className="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-mono border border-neutral-300 bg-neutral-50 text-black hover:border-black hover:bg-neutral-100 transition-colors rounded"
                  title="Click to view clause in document"
                >
                  <FileText className="w-3 h-3 text-neutral-700" />
                  <span>
                    {clause
                      ? `Clause #${clause.clause_index + 1}: ${clause.clause_type?.replace(/_/g, " ").toUpperCase()}`
                      : `Clause ${clauseId.slice(0, 8)}`}
                  </span>
                </button>
              );
            })}
          </div>
        </div>
      )}

      {!isUser && message.retrieved_clauses && message.retrieved_clauses.length > 0 && (
        <div className="mt-3 pt-2">
          <button
            onClick={() => setShowRetrieved(!showRetrieved)}
            className="text-[11px] font-mono text-neutral-500 hover:text-black flex items-center gap-1 transition-colors"
          >
            {showRetrieved ? (
              <>
                <ChevronUp className="w-3 h-3" />
                <span>Hide {message.retrieved_clauses.length} vector-retrieved clauses</span>
              </>
            ) : (
              <>
                <ChevronDown className="w-3 h-3" />
                <span>Inspect {message.retrieved_clauses.length} vector-retrieved clauses passed to AI Copilot</span>
              </>
            )}
          </button>

          {showRetrieved && (
            <div className="mt-2 space-y-2 border-l-2 border-neutral-300 pl-3 pt-1">
              {message.retrieved_clauses.map((rc, idx) => (
                <div key={idx} className="bg-neutral-50 border border-neutral-200 p-2.5 text-xs rounded">
                  <div className="flex items-center justify-between text-[10px] font-mono text-neutral-600 mb-1">
                    <span>Clause #{rc.clause_index + 1} ({rc.clause_type})</span>
                    <span>Distance: {rc.distance?.toFixed(4) || "N/A"}</span>
                  </div>
                  <p className="text-neutral-800 text-xs italic font-serif line-clamp-3">
                    "{rc.redacted_text}"
                  </p>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default ChatMessage;
