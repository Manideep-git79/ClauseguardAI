import React, { useState, useEffect, useMemo } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import {
  PIPELINE_CATEGORIES,
  PIPELINE_STAGES,
  AUXILIARY_BRANCH_STAGES,
  RAG_BRANCH_STAGES,
} from "../data/architectureFlowData";
import { FlowchartNode } from "../components/architecture/FlowchartNode";
import {
  GitFork,
  CheckCircle2,
  Layers,
  Search,
  Maximize2,
  Minimize2,
  ArrowUp,
  FileCode,
  ShieldCheck,
  ShieldAlert,
  Sparkles,
  Bot,
  Binary,
  ArrowDown,
  Info,
} from "lucide-react";

export function ArchitectureFlow({ onNavigate }) {
  const location = useLocation();
  const navigate = useNavigate();

  // Combine all stages for reference lookups
  const allStages = useMemo(() => {
    return [
      ...PIPELINE_STAGES,
      ...AUXILIARY_BRANCH_STAGES,
      ...RAG_BRANCH_STAGES,
    ];
  }, []);

  // State
  const [activeCategory, setActiveCategory] = useState("ALL");
  const [searchQuery, setSearchQuery] = useState("");
  const [expandedNodes, setExpandedNodes] = useState(() => {
    // Stage 01 and Stage 08 expanded by default for instant architectural clarity
    return new Set(["stage-01-upload", "stage-08-eligibility-gate"]);
  });

  // Handle URL hash / deep linking
  useEffect(() => {
    const hash = location.hash.replace("#", "");
    if (hash) {
      setExpandedNodes((prev) => new Set([...prev, hash]));
      setTimeout(() => {
        const el = document.getElementById(hash);
        if (el) {
          el.scrollIntoView({ behavior: "smooth", block: "center" });
        }
      }, 150);
    }
  }, [location.hash]);

  // Toggle single node expansion
  const toggleNode = (stageId) => {
    setExpandedNodes((prev) => {
      const next = new Set(prev);
      if (next.has(stageId)) {
        next.delete(stageId);
      } else {
        next.add(stageId);
      }
      return next;
    });
  };

  // Expand / collapse all
  const expandAll = () => {
    const allIds = allStages.map((s) => s.id);
    setExpandedNodes(new Set(allIds));
  };

  const collapseAll = () => {
    setExpandedNodes(new Set());
  };

  // Navigate to stage with smooth scroll
  const handleNavigateStage = (stageId) => {
    setExpandedNodes((prev) => new Set([...prev, stageId]));
    setTimeout(() => {
      const el = document.getElementById(stageId);
      if (el) {
        el.scrollIntoView({ behavior: "smooth", block: "center" });
      }
    }, 100);
  };

  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  // Filter stages based on category and search query
  const filteredMainStages = useMemo(() => {
    return PIPELINE_STAGES.filter((stage) => {
      const matchesCategory =
        activeCategory === "ALL" ||
        stage.category === activeCategory ||
        (activeCategory === "INGESTION" && ["INGESTION"].includes(stage.category)) ||
        (activeCategory === "EXTRACTION" && ["EXTRACTION"].includes(stage.category)) ||
        (activeCategory === "SEGMENTATION" && ["SEGMENTATION"].includes(stage.category)) ||
        (activeCategory === "ML_RISK" && ["CLASSIFICATION", "RISK_ANALYSIS"].includes(stage.category)) ||
        (activeCategory === "PERSISTENCE" && ["PERSISTENCE", "PRESENTATION"].includes(stage.category));

      if (!matchesCategory && activeCategory !== "ALL") return false;

      if (!searchQuery.trim()) return true;

      const q = searchQuery.toLowerCase();
      const matchTitle = stage.title.toLowerCase().includes(q);
      const matchSummary = stage.summary.toLowerCase().includes(q);
      const matchPlain = stage.plainLanguageExplanation?.toLowerCase().includes(q);
      const matchCode = stage.codeReferences?.some(
        (ref) =>
          ref.file.toLowerCase().includes(q) ||
          ref.symbol.toLowerCase().includes(q) ||
          ref.explanation.toLowerCase().includes(q)
      );

      return matchTitle || matchSummary || matchPlain || matchCode;
    });
  }, [activeCategory, searchQuery]);

  const filteredAuxStages = useMemo(() => {
    if (activeCategory !== "ALL" && activeCategory !== "BRANCHES") return [];
    if (!searchQuery.trim()) return AUXILIARY_BRANCH_STAGES;
    const q = searchQuery.toLowerCase();
    return AUXILIARY_BRANCH_STAGES.filter(
      (s) =>
        s.title.toLowerCase().includes(q) ||
        s.summary.toLowerCase().includes(q) ||
        s.codeReferences?.some((r) => r.file.toLowerCase().includes(q) || r.symbol.toLowerCase().includes(q))
    );
  }, [activeCategory, searchQuery]);

  const filteredRagStages = useMemo(() => {
    if (activeCategory !== "ALL" && activeCategory !== "BRANCHES") return [];
    if (!searchQuery.trim()) return RAG_BRANCH_STAGES;
    const q = searchQuery.toLowerCase();
    return RAG_BRANCH_STAGES.filter(
      (s) =>
        s.title.toLowerCase().includes(q) ||
        s.summary.toLowerCase().includes(q) ||
        s.codeReferences?.some((r) => r.file.toLowerCase().includes(q) || r.symbol.toLowerCase().includes(q))
    );
  }, [activeCategory, searchQuery]);

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 py-6 space-y-10">
      {/* -------------------------------------------------------------------- */}
      {/* 1. HEADER SECTION (Restrained Monochrome Editorial Style)            */}
      {/* -------------------------------------------------------------------- */}
      <header className="border-b border-neutral-300 pb-8 text-left space-y-3">
        <div className="flex items-center gap-2 text-xs font-mono tracking-widest text-neutral-500 uppercase font-semibold">
          <span>CLAUSEGUARD AI</span>
          <span>•</span>
          <span>SYSTEM ARCHITECTURE & EXECUTION TRACE</span>
        </div>

        <h1 className="font-serif text-3xl sm:text-4xl font-bold tracking-tight text-neutral-900 leading-tight">
          How a Document Becomes a Risk Analysis
        </h1>

        <p className="font-sans text-sm sm:text-base text-neutral-600 max-w-2xl leading-relaxed">
          Follow a document through extraction, clause segmentation, machine-learning analysis, risk scoring,
          and evidence-grounded AI assistance. Select any step to inspect the verified implementation, source
          lines, and decision logic.
        </p>

        {/* ------------------------------------------------------------------ */}
        {/* 2. COMPACT UNDERSTATED LEGEND                                      */}
        {/* ------------------------------------------------------------------ */}
        <div className="pt-4 flex flex-wrap items-center gap-y-2 gap-x-5 text-xs font-mono text-neutral-600">
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 bg-neutral-100 border border-neutral-400 rounded-sm inline-block"></span>
            <span>Processing Stage</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 bg-neutral-100 border border-neutral-800 rounded-sm inline-block flex items-center justify-center">
              <GitFork className="w-2.5 h-2.5 text-neutral-800" />
            </span>
            <span>Decision / Branching Point</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 bg-neutral-300 rounded-full inline-block"></span>
            <span>Exclusion / Gating Filter</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 bg-neutral-900 rounded-sm inline-block"></span>
            <span>Active / Expanded</span>
          </div>
          <div className="flex items-center gap-1.5">
            <CheckCircle2 className="w-3.5 h-3.5 text-neutral-700" />
            <span>Verified Code Reference</span>
          </div>
        </div>
      </header>

      {/* -------------------------------------------------------------------- */}
      {/* 3. STICKY CONTROLS & STAGE NAVIGATOR                                 */}
      {/* -------------------------------------------------------------------- */}
      <div className="sticky top-16 z-30 bg-white/95 backdrop-blur-sm border border-neutral-200 p-3 sm:p-4 rounded-lg shadow-xs space-y-3">
        <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3">
          {/* Search Bar */}
          <div className="relative flex-1">
            <Search className="w-4 h-4 text-neutral-400 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search stages, files, or symbols (e.g. block_segmenter, DistilBERT, OCR)..."
              className="w-full pl-9 pr-3 py-1.5 text-xs font-mono border border-neutral-300 rounded focus:outline-none focus:border-black bg-white"
            />
            {searchQuery && (
              <button
                onClick={() => setSearchQuery("")}
                className="absolute right-2.5 top-1/2 -translate-y-1/2 text-neutral-400 hover:text-black text-xs font-mono"
              >
                ×
              </button>
            )}
          </div>

          {/* Quick Actions */}
          <div className="flex items-center gap-2 justify-end text-xs font-mono">
            <button
              onClick={expandAll}
              className="px-2.5 py-1.5 border border-neutral-300 hover:border-black bg-neutral-50 hover:bg-neutral-100 rounded text-neutral-700 hover:text-black flex items-center gap-1 transition-colors"
              title="Expand all stage details"
            >
              <Maximize2 className="w-3 h-3" />
              <span className="hidden sm:inline">Expand All</span>
            </button>
            <button
              onClick={collapseAll}
              className="px-2.5 py-1.5 border border-neutral-300 hover:border-black bg-neutral-50 hover:bg-neutral-100 rounded text-neutral-700 hover:text-black flex items-center gap-1 transition-colors"
              title="Collapse all stage details"
            >
              <Minimize2 className="w-3 h-3" />
              <span className="hidden sm:inline">Collapse</span>
            </button>
            <button
              onClick={scrollToTop}
              className="px-2.5 py-1.5 border border-neutral-300 hover:border-black bg-neutral-50 hover:bg-neutral-100 rounded text-neutral-700 hover:text-black flex items-center gap-1 transition-colors"
              title="Scroll to top"
            >
              <ArrowUp className="w-3 h-3" />
              <span className="hidden sm:inline">Top</span>
            </button>
          </div>
        </div>

        {/* Category Filter Pills */}
        <div className="flex items-center gap-1.5 overflow-x-auto pb-1 text-xs font-mono scrollbar-none">
          {PIPELINE_CATEGORIES.map((cat) => (
            <button
              key={cat.id}
              onClick={() => setActiveCategory(cat.id)}
              className={`px-2.5 py-1 rounded whitespace-nowrap transition-colors border ${
                activeCategory === cat.id
                  ? "bg-black text-white border-black font-semibold"
                  : "bg-white text-neutral-600 border-neutral-200 hover:border-neutral-400 hover:text-black"
              }`}
            >
              {cat.label}
            </button>
          ))}
        </div>
      </div>

      {/* -------------------------------------------------------------------- */}
      {/* 4. MAIN VERTICAL FLOWCHART (Stages 01 to 15)                          */}
      {/* -------------------------------------------------------------------- */}
      <section className="space-y-1">
        <div className="flex items-center justify-between pb-3 border-b border-neutral-200 text-xs font-mono text-neutral-500">
          <span className="uppercase tracking-wider font-semibold text-neutral-700">
            Primary Pipeline: Ingestion → Classification → Risk Scoring ({filteredMainStages.length} of 15)
          </span>
          <span>End-to-End Traced</span>
        </div>

        <div className="pt-4">
          {filteredMainStages.length > 0 ? (
            filteredMainStages.map((stage, idx) => (
              <FlowchartNode
                key={stage.id}
                stage={stage}
                isExpanded={expandedNodes.has(stage.id)}
                onToggle={() => toggleNode(stage.id)}
                onNavigateStage={handleNavigateStage}
                isLast={idx === filteredMainStages.length - 1}
                allStages={allStages}
              />
            ))
          ) : (
            <div className="text-center py-12 border border-dashed border-neutral-300 rounded-lg text-neutral-500 font-mono text-xs">
              No pipeline stages match the active filter or search query.
            </div>
          )}
        </div>
      </section>

      {/* -------------------------------------------------------------------- */}
      {/* 5. SEPARATE DOWNSTREAM BRANCHES (Auxiliary Intelligence & RAG Chat)  */}
      {/* -------------------------------------------------------------------- */}
      {(filteredAuxStages.length > 0 || filteredRagStages.length > 0) && (
        <section className="pt-8 border-t-2 border-neutral-300 space-y-8">
          <div className="text-left space-y-1.5">
            <div className="text-xs font-mono uppercase tracking-widest text-neutral-500 font-semibold">
              Downstream Branches
            </div>
            <h2 className="font-serif text-2xl font-bold text-neutral-900">
              Specialized Parallel Pipelines
            </h2>
            <p className="text-xs sm:text-sm text-neutral-600 font-sans leading-relaxed">
              These processes run in coordination with the primary pipeline. They share the normalized text and
              extracted clauses without complicating the core classification and risk calculation lifecycle.
            </p>
          </div>

          {/* Branch A: Auxiliary Document Intelligence */}
          {filteredAuxStages.length > 0 && (
            <div className="border border-neutral-300 rounded-xl p-5 bg-neutral-50/50 space-y-4">
              <div className="flex items-center justify-between border-b border-neutral-200 pb-3">
                <div className="flex items-center gap-2">
                  <ShieldCheck className="w-4 h-4 text-neutral-800" />
                  <h3 className="font-serif font-bold text-base text-neutral-900">
                    Branch A: Auxiliary Document Intelligence
                  </h3>
                </div>
                <span className="text-[11px] font-mono text-neutral-500">
                  Runs during Stage 04 & 09
                </span>
              </div>
              <p className="text-xs text-neutral-600 font-sans">
                Protects user privacy through Presidio PII pseudonymization, scans for visual tampering / prompt injection,
                and parses operational notice windows and monetary penalties.
              </p>
              <div className="space-y-3 pt-2">
                {filteredAuxStages.map((stage) => (
                  <FlowchartNode
                    key={stage.id}
                    stage={stage}
                    isExpanded={expandedNodes.has(stage.id)}
                    onToggle={() => toggleNode(stage.id)}
                    onNavigateStage={handleNavigateStage}
                    isLast={true}
                    allStages={allStages}
                  />
                ))}
              </div>
            </div>
          )}

          {/* Branch B: RAG Semantic Indexing & Legal Copilot */}
          {filteredRagStages.length > 0 && (
            <div className="border border-neutral-300 rounded-xl p-5 bg-neutral-50/50 space-y-4">
              <div className="flex items-center justify-between border-b border-neutral-200 pb-3">
                <div className="flex items-center gap-2">
                  <Bot className="w-4 h-4 text-neutral-800" />
                  <h3 className="font-serif font-bold text-base text-neutral-900">
                    Branch B: RAG Indexing & Legal AI Copilot
                  </h3>
                </div>
                <span className="text-[11px] font-mono text-neutral-500">
                  Consumes Stored Clauses (Stage 14)
                </span>
              </div>
              <p className="text-xs text-neutral-600 font-sans">
                Generates 768-dim embeddings in PostgreSQL pgvector, normalizes user intent (preventing false "not found" rejections),
                and executes weighted Reciprocal Rank Fusion before synthesizing grounded legal answers.
              </p>
              <div className="space-y-3 pt-2">
                {filteredRagStages.map((stage) => (
                  <FlowchartNode
                    key={stage.id}
                    stage={stage}
                    isExpanded={expandedNodes.has(stage.id)}
                    onToggle={() => toggleNode(stage.id)}
                    onNavigateStage={handleNavigateStage}
                    isLast={true}
                    allStages={allStages}
                  />
                ))}
              </div>
            </div>
          )}
        </section>
      )}

      {/* -------------------------------------------------------------------- */}
      {/* 6. VERIFICATION METHODOLOGY & FOOTER                                 */}
      {/* -------------------------------------------------------------------- */}
      <footer className="pt-10 border-t border-neutral-200 text-xs font-mono text-neutral-500 space-y-3">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2">
          <div>
            <strong className="text-neutral-800">Verification Source: </strong>
            Direct ast/inspect AST trace on ClauseGuard AI repository.
          </div>
          <div>
            <span>15 Core Stages</span> • <span>3 Auxiliary</span> • <span>4 RAG Stages</span>
          </div>
        </div>
        <p className="text-[11px] text-neutral-600 leading-normal">
          Every file path, function symbol, and line range on this page is extracted from the active codebase.
          Gating filters (e.g., non-substantive structural exclusions in <code className="bg-neutral-100 px-1 py-0.5 rounded text-neutral-800">clause_eligibility_gate.py</code>)
          and consensus thresholds are verified against current production files.
        </p>
      </footer>
    </div>
  );
}

export default ArchitectureFlow;
