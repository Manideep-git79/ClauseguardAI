import React, { useState, useMemo } from "react";
import { ClauseCard } from "./ClauseCard";
import { Search, ArrowUpDown, Filter, ChevronDown, ChevronUp, ShieldAlert, FileText, Info } from "lucide-react";

export function ClauseList({ clauses = [], selectedClauseId, onSelectClause }) {
  const [filter, setFilter] = useState("all");
  const [search, setSearch] = useState("");
  const [sortBy, setSortBy] = useState("index");
  const [showExcluded, setShowExcluded] = useState(false);

  // Distinguish substantive clauses from non-substantive blocks
  const { substantiveClauses, excludedBlocks, excludedBreakdown } = useMemo(() => {
    const substantive = [];
    const excluded = [];
    const breakdown = {};

    clauses.forEach((c) => {
      const isSub = c.is_substantive !== false && (c.content_type === "SUBSTANTIVE_CLAUSE" || c.content_type === "DEFINITION");
      if (isSub) {
        substantive.push(c);
      } else {
        excluded.push(c);
        const type = c.content_type || "OTHER_NON_SUBSTANTIVE";
        breakdown[type] = (breakdown[type] || 0) + 1;
      }
    });

    return { substantiveClauses: substantive, excludedBlocks: excluded, excludedBreakdown: breakdown };
  }, [clauses]);

  const filteredClauses = useMemo(() => {
    return substantiveClauses
      .filter((c) => {
        if (filter === "needs_review") {
          if (!c.human_review && c.favorability_label !== "needs_review" && c.clause_type !== "UNCERTAIN" && c.clause_type !== "UNCLASSIFIED") return false;
        } else if (filter === "fair") {
          if (c.favorability_label !== "fair" || c.human_review || c.clause_type === "UNCERTAIN" || c.clause_type === "UNCLASSIFIED") return false;
        } else if (filter !== "all" && c.favorability_label !== filter) {
          return false;
        }

        if (search) {
          const q = search.toLowerCase();
          const matchType = (c.clause_type || "").toLowerCase().includes(q);
          const matchText = (c.redacted_text || "").toLowerCase().includes(q);
          if (!matchType && !matchText) return false;
        }
        return true;
      })
      .sort((a, b) => {
        if (sortBy === "risk") {
          return (b.risk_score || 0) - (a.risk_score || 0);
        }
        return a.clause_index - b.clause_index;
      });
  }, [substantiveClauses, filter, search, sortBy]);

  const formatTypeName = (rawType) => {
    if (!rawType) return "Other";
    return rawType
      .replace(/_/g, " ")
      .toLowerCase()
      .replace(/\b\w/g, (c) => c.toUpperCase());
  };

  return (
    <div className="space-y-6">
      {/* Search and Filters Bar */}
      <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3 border border-neutral-200 bg-white p-4 shadow-sm">
        <div className="relative flex-1">
          <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-neutral-400" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search substantive legal clauses or keywords..."
            className="w-full bg-white border border-neutral-300 pl-9 pr-3 py-2 text-xs text-black placeholder-neutral-400 focus:outline-none focus:border-black font-sans"
          />
        </div>

        <div className="flex items-center gap-1 overflow-x-auto">
          {["all", "unfavorable", "needs_review", "fair"].map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`px-3 py-1.5 text-[11px] font-mono uppercase tracking-wider border whitespace-nowrap transition-colors ${
                filter === f
                  ? "border-black bg-black text-white font-bold"
                  : "border-neutral-300 bg-white text-neutral-700 hover:text-black hover:border-black"
              }`}
            >
              {f.replace("_", " ")}
            </button>
          ))}

          <button
            onClick={() => setSortBy(sortBy === "index" ? "risk" : "index")}
            className="px-3 py-1.5 text-[11px] font-mono uppercase tracking-wider border border-neutral-300 bg-white text-neutral-700 hover:text-black hover:border-black flex items-center gap-1.5 transition-colors"
            title="Toggle sort order"
          >
            <ArrowUpDown className="w-3 h-3" />
            {sortBy === "risk" ? "Risk Score" : "Document Order"}
          </button>
        </div>
      </div>

      {/* Primary Substantive Clauses List */}
      {filteredClauses.length > 0 ? (
        <div className="space-y-3">
          {filteredClauses.map((clause, idx) => (
            <ClauseCard
              key={clause.id}
              clause={clause}
              displayIndex={idx + 1}
              isSelected={selectedClauseId === clause.id}
              onSelect={() => onSelectClause(clause.id)}
            />
          ))}
        </div>
      ) : (
        <div className="border border-neutral-200 bg-white p-8 text-center text-neutral-500 text-xs font-mono shadow-sm">
          No substantive legal clauses match the selected search or filter criteria.
        </div>
      )}

      {/* Collapsed Diagnostic Section: Non-substantive Content Excluded */}
      {excludedBlocks.length > 0 && (
        <div className="border border-neutral-300 bg-neutral-50 shadow-sm transition-all">
          <button
            onClick={() => setShowExcluded(!showExcluded)}
            className="w-full px-5 py-4 flex items-center justify-between text-left hover:bg-neutral-100 transition-colors"
          >
            <div className="flex items-center gap-3">
              <ShieldAlert className="w-4 h-4 text-neutral-500" />
              <div>
                <span className="text-xs font-mono uppercase font-bold text-neutral-800 tracking-wider">
                  Non-Substantive Content Excluded ({excludedBlocks.length} Blocks)
                </span>
                <span className="block text-[11px] font-sans text-neutral-500 mt-0.5">
                  Front matter, metadata, headers, footers, and product descriptions filtered from risk scoring
                </span>
              </div>
            </div>
            <div className="flex items-center gap-2 text-xs font-mono text-neutral-600">
              <span>{showExcluded ? "Collapse Details" : "View Breakdown"}</span>
              {showExcluded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </div>
          </button>

          {showExcluded && (
            <div className="border-t border-neutral-200 p-5 space-y-4 bg-white">
              {/* Summary Counts by Content Type */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                {Object.entries(excludedBreakdown).map(([cType, count]) => (
                  <div key={cType} className="border border-neutral-200 p-2.5 bg-neutral-50">
                    <span className="text-[10px] font-mono text-neutral-500 uppercase block font-semibold">
                      {formatTypeName(cType)}
                    </span>
                    <span className="text-lg font-mono font-bold text-black mt-0.5 block">
                      {count}
                    </span>
                  </div>
                ))}
              </div>

              <div className="text-[11px] font-sans text-neutral-600 bg-neutral-100 p-3 border border-neutral-200 flex items-start gap-2">
                <Info className="w-4 h-4 text-neutral-500 flex-shrink-0 mt-0.5" />
                <span>
                  The items below were evaluated by the Clause Eligibility Gate and identified as non-substantive.
                  They do not enter DistilBERT classification or legal risk scoring, and strictly carry 0 risk.
                </span>
              </div>

              {/* Excluded Items Preview List */}
              <div className="space-y-2 max-h-96 overflow-y-auto pr-1">
                {excludedBlocks.map((b, idx) => (
                  <div
                    key={b.id || idx}
                    className="border border-neutral-200 p-3 bg-neutral-50/50 hover:bg-neutral-100 transition-colors"
                  >
                    <div className="flex items-center justify-between gap-2 mb-1.5">
                      <div className="flex items-center gap-2">
                        <span className="text-[10px] font-mono uppercase px-2 py-0.5 border border-neutral-300 bg-white font-semibold text-neutral-700">
                          {b.content_type || "NON_SUBSTANTIVE"}
                        </span>
                        <span className="text-[10px] font-mono text-neutral-500">
                          Risk: 0.0 • NOT_APPLICABLE
                        </span>
                      </div>
                      {b.filter_reason && (
                        <span className="text-[10px] font-mono text-neutral-400 italic">
                          {b.filter_reason}
                        </span>
                      )}
                    </div>
                    <p className="text-xs font-mono text-neutral-700 line-clamp-2">
                      {b.redacted_text || b.text}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
