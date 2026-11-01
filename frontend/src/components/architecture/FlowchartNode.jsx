import React, { useState } from "react";
import {
  ChevronDown,
  ChevronUp,
  FileCode,
  ArrowRight,
  CheckCircle2,
  AlertTriangle,
  GitFork,
  Copy,
  Check,
  ExternalLink,
  ShieldCheck,
  ShieldAlert,
  HelpCircle,
} from "lucide-react";
import { MiniDiagram } from "./MiniDiagram";

export function FlowchartNode({
  stage,
  isExpanded,
  onToggle,
  onNavigateStage,
  isLast = false,
  allStages = [],
}) {
  const [copiedPath, setCopiedPath] = useState(null);

  const handleCopy = (path, e) => {
    e.stopPropagation();
    navigator.clipboard.writeText(path);
    setCopiedPath(path);
    setTimeout(() => setCopiedPath(null), 2000);
  };

  const nextStage = allStages.find((s) => s.id === stage.nextStageId);

  return (
    <div id={stage.id} className="relative group scroll-mt-28">
      {/* Node Container */}
      <div
        className={`border transition-all duration-200 rounded-lg bg-white ${
          isExpanded
            ? "border-black shadow-md ring-1 ring-black/5"
            : "border-neutral-300 hover:border-neutral-500 shadow-sm"
        }`}
      >
        {/* Node Header (Clickable / Accessible) */}
        <button
          type="button"
          onClick={onToggle}
          aria-expanded={isExpanded}
          className="w-full text-left p-4 sm:p-5 flex items-start justify-between gap-4 cursor-pointer focus:outline-none focus-visible:ring-2 focus-visible:ring-black rounded-lg"
        >
          <div className="flex items-start gap-3.5 min-w-0">
            {/* Stage Number Badge */}
            <div
              className={`w-9 h-9 flex-shrink-0 flex items-center justify-center rounded font-mono text-xs font-bold transition-colors ${
                isExpanded
                  ? "bg-black text-white"
                  : stage.isDecision
                  ? "bg-neutral-100 text-neutral-900 border border-neutral-800"
                  : "bg-neutral-100 text-neutral-800 border border-neutral-300 group-hover:border-neutral-400"
              }`}
            >
              {stage.isDecision ? <GitFork className="w-4 h-4" /> : stage.stageNumber}
            </div>

            {/* Title & Summary */}
            <div className="min-w-0 flex-1">
              <div className="flex flex-wrap items-center gap-2 mb-1">
                <span className="text-[10px] font-mono uppercase tracking-wider text-neutral-500 font-semibold">
                  Stage {stage.stageNumber} • {stage.category}
                </span>

                {stage.isDecision && (
                  <span className="text-[10px] font-mono uppercase tracking-wider px-2 py-0.5 bg-neutral-100 text-neutral-800 border border-neutral-400 rounded font-semibold">
                    Decision Gate
                  </span>
                )}

                <span className="text-[10px] font-mono px-2 py-0.5 bg-neutral-100 text-neutral-700 border border-neutral-200 rounded">
                  {stage.status || "Implemented & Verified"}
                </span>
              </div>

              <h3 className="font-serif text-base sm:text-lg font-bold text-neutral-900 leading-tight">
                {stage.title}
              </h3>

              <p className="text-xs text-neutral-600 mt-1 line-clamp-2 font-sans">
                {stage.summary}
              </p>
            </div>
          </div>

          {/* Disclosure Chevron */}
          <div className="flex-shrink-0 text-neutral-400 group-hover:text-black mt-1 p-1">
            {isExpanded ? (
              <ChevronUp className="w-5 h-5 text-black" />
            ) : (
              <ChevronDown className="w-5 h-5" />
            )}
          </div>
        </button>

        {/* Expanded Detailed Section */}
        {isExpanded && (
          <div className="border-t border-neutral-200 px-4 sm:px-6 py-5 bg-neutral-50/40 rounded-b-lg space-y-6 text-sm">
            {/* Section A: Plain Language Explanation */}
            <div>
              <div className="text-[11px] font-mono uppercase tracking-wider text-neutral-500 font-bold mb-1.5 flex items-center gap-1.5">
                <span>01. What Happens Here?</span>
              </div>
              <p className="text-neutral-800 leading-relaxed font-sans text-xs sm:text-sm">
                {stage.plainLanguageExplanation}
              </p>
            </div>

            {/* Section B: Inputs & Outputs */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Inputs */}
              <div className="border border-neutral-200 bg-white p-3.5 rounded">
                <div className="text-[10px] font-mono uppercase tracking-wider text-neutral-500 font-bold mb-2">
                  Inputs Received
                </div>
                <div className="space-y-2">
                  {stage.inputs && stage.inputs.length > 0 ? (
                    stage.inputs.map((inp, idx) => (
                      <div key={idx} className="text-xs">
                        <div className="flex items-center justify-between">
                          <span className="font-mono font-semibold text-neutral-900">{inp.name}</span>
                          <span className="text-[10px] font-mono text-neutral-600 bg-neutral-100 px-1.5 py-0.5 rounded">
                            {inp.type}
                          </span>
                        </div>
                        <div className="text-[11px] text-neutral-600 mt-0.5">{inp.description}</div>
                      </div>
                    ))
                  ) : (
                    <div className="text-xs text-neutral-600">None specified</div>
                  )}
                </div>
              </div>

              {/* Outputs */}
              <div className="border border-neutral-200 bg-white p-3.5 rounded">
                <div className="text-[10px] font-mono uppercase tracking-wider text-neutral-500 font-bold mb-2">
                  Outputs Produced
                </div>
                <div className="space-y-2">
                  {stage.outputs && stage.outputs.length > 0 ? (
                    stage.outputs.map((out, idx) => (
                      <div key={idx} className="text-xs">
                        <div className="flex items-center justify-between">
                          <span className="font-mono font-semibold text-neutral-900">{out.name}</span>
                          <span className="text-[10px] font-mono text-neutral-600 bg-neutral-100 px-1.5 py-0.5 rounded">
                            {out.type}
                          </span>
                        </div>
                        <div className="text-[11px] text-neutral-600 mt-0.5">{out.description}</div>
                      </div>
                    ))
                  ) : (
                    <div className="text-xs text-neutral-600">Continuous in-memory updates</div>
                  )}
                </div>
              </div>
            </div>

            {/* Failure / Edge Cases (If any) */}
            {stage.failurePaths && stage.failurePaths.length > 0 && (
              <div className="border border-neutral-200 bg-white p-3.5 rounded">
                <div className="text-[10px] font-mono uppercase tracking-wider text-neutral-500 font-bold mb-2">
                  Edge Cases & Failure Behavior
                </div>
                <div className="space-y-1.5 text-xs">
                  {stage.failurePaths.map((fp, idx) => (
                    <div key={idx} className="flex flex-col sm:flex-row sm:items-center justify-between gap-1 py-1 border-b border-neutral-100 last:border-b-0">
                      <span className="text-neutral-700">
                        <strong className="text-neutral-900">Condition:</strong> {fp.condition}
                      </span>
                      <span className="text-neutral-600 font-mono text-[11px]">
                        {fp.result} {fp.httpStatus && `(${fp.httpStatus})`}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Decision Branches (If decision node) */}
            {stage.decisionBranches && stage.decisionBranches.length > 0 && (
              <div className="border border-neutral-300 bg-white p-4 rounded-lg">
                <div className="text-[11px] font-mono uppercase tracking-wider text-neutral-700 font-bold mb-3 flex items-center gap-1.5">
                  <GitFork className="w-3.5 h-3.5 text-neutral-900" />
                  <span>Verified Decision Branches</span>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {stage.decisionBranches.map((br, idx) => (
                    <div key={idx} className="border border-neutral-200 p-3 rounded bg-neutral-50/50">
                      <div className="font-mono text-xs font-bold text-neutral-900">{br.name}</div>
                      <div className="text-xs text-neutral-700 mt-1">
                        <strong className="text-neutral-900">Condition:</strong> {br.condition}
                      </div>
                      <div className="text-xs text-neutral-600 mt-1">
                        <strong className="text-neutral-900">Effect:</strong> {br.effect}
                      </div>
                      <div className="text-[10px] font-mono text-neutral-600 bg-white border border-neutral-200 p-1.5 rounded mt-2">
                        {br.codeBehavior}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Section D: Native Visual Explanation */}
            {stage.miniDiagramType && (
              <div>
                <div className="text-[11px] font-mono uppercase tracking-wider text-neutral-500 font-bold mb-1.5">
                  02. Process Diagram
                </div>
                <MiniDiagram type={stage.miniDiagramType} />
              </div>
            )}

            {/* Section E: Concrete Real-World Example (If provided) */}
            {stage.example && (
              <div className="border border-neutral-200 bg-white p-3.5 rounded text-xs">
                <div className="text-[10px] font-mono uppercase tracking-wider text-neutral-500 font-bold mb-1">
                  Concrete Execution Example
                </div>
                <div className="font-mono text-neutral-800 bg-neutral-100 p-2 rounded mb-1 text-[11px]">
                  <strong>Input:</strong> {stage.example.input}
                </div>
                <div className="font-mono text-neutral-800 bg-neutral-50 p-2 border border-neutral-200 rounded text-[11px]">
                  <strong>Output:</strong> {stage.example.output}
                </div>
              </div>
            )}

            {/* Section F: Verified Code References */}
            <div>
              <div className="text-[11px] font-mono uppercase tracking-wider text-neutral-500 font-bold mb-2 flex items-center justify-between">
                <span>03. Verified Source Code References</span>
                <span className="text-[10px] text-neutral-600 font-normal">
                  Inspected via Python inspect & git
                </span>
              </div>

              <div className="space-y-3">
                {stage.codeReferences && stage.codeReferences.map((ref, idx) => (
                  <div
                    key={idx}
                    className="border border-neutral-300 bg-white rounded-lg overflow-hidden shadow-xs"
                  >
                    {/* Reference Header */}
                    <div className="bg-neutral-100/80 px-3.5 py-2 border-b border-neutral-200 flex flex-wrap items-center justify-between gap-2">
                      <div className="flex items-center gap-2 min-w-0">
                        <FileCode className="w-3.5 h-3.5 text-neutral-700 flex-shrink-0" />
                        <span className="font-mono text-xs font-semibold text-neutral-900 truncate">
                          {ref.file}
                        </span>
                        <button
                          type="button"
                          onClick={(e) => handleCopy(ref.file, e)}
                          title="Copy file path"
                          className="text-neutral-500 hover:text-black transition-colors"
                        >
                          {copiedPath === ref.file ? (
                            <Check className="w-3 h-3 text-neutral-900" />
                          ) : (
                            <Copy className="w-3 h-3" />
                          )}
                        </button>
                      </div>

                      <div className="flex items-center gap-2 font-mono text-[11px]">
                        <span className="text-neutral-800 font-semibold">{ref.symbol}</span>
                        <span className="bg-white border border-neutral-300 px-1.5 py-0.5 rounded text-neutral-700">
                          {ref.lines}
                        </span>
                      </div>
                    </div>

                    {/* Monospace Code Excerpt */}
                    <div className="p-3 bg-neutral-900 text-neutral-100 font-mono text-[11px] leading-relaxed overflow-x-auto">
                      <pre className="whitespace-pre">{ref.snippet}</pre>
                    </div>

                    {/* Excerpt Explanation & Next Call */}
                    <div className="p-3 bg-neutral-50 text-xs border-t border-neutral-200 space-y-1">
                      <p className="text-neutral-700 font-sans">
                        <strong className="text-neutral-900 font-mono text-[11px]">Implementation: </strong>
                        {ref.explanation}
                      </p>
                      {ref.nextCalled && (
                        <div className="text-[11px] text-neutral-600 font-mono pt-1">
                          <strong className="text-neutral-800">Calls Next: </strong>
                          {ref.nextCalled}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Limitations or Gaps (if any) */}
            {stage.limitations && (
              <div className="border border-neutral-300 bg-neutral-100/60 p-3 rounded text-xs text-neutral-700">
                <div className="font-mono font-bold text-neutral-900 text-[10px] uppercase mb-0.5">
                  Design Note / Current Implementation Limits
                </div>
                {stage.limitations}
              </div>
            )}

            {/* Navigation to Next Step */}
            {stage.nextStageId && (
              <div className="pt-2 border-t border-neutral-200 flex items-center justify-between">
                <div className="text-xs text-neutral-500 font-mono">
                  Continuous Pipeline Link:
                </div>
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    onNavigateStage(stage.nextStageId);
                  }}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-mono font-semibold bg-black text-white hover:bg-neutral-800 rounded transition-colors"
                >
                  <span>Next: {nextStage ? nextStage.title : stage.nextStageId}</span>
                  <ArrowRight className="w-3.5 h-3.5" />
                </button>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Downward Connector Arrow (Only between sequential vertical nodes) */}
      {!isLast && (
        <div className="w-full flex justify-center py-2" aria-hidden="true">
          <div className="flex flex-col items-center">
            <div className="w-px h-5 bg-neutral-300"></div>
            <div className="w-0 h-0 border-l-[4px] border-l-transparent border-r-[4px] border-r-transparent border-t-[6px] border-t-neutral-400 -mt-0.5"></div>
          </div>
        </div>
      )}
    </div>
  );
}
