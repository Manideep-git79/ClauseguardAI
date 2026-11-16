import React, { useState } from "react";
import {
  Code2,
  FileCode,
  MapPin,
  Star,
  Lightbulb,
  Image as ImageIcon,
  Check,
  Copy,
  ArrowRight,
  Sparkles,
  ChevronRight,
  Terminal,
} from "lucide-react";

export function CodeReviewCard({ stage, isExpanded = true, onToggle }) {
  const [copiedPath, setCopiedPath] = useState(false);
  const [copiedCode, setCopiedCode] = useState(false);

  const handleCopyPath = (e) => {
    e.stopPropagation();
    navigator.clipboard.writeText(stage.fileLocation);
    setCopiedPath(true);
    setTimeout(() => setCopiedPath(false), 2000);
  };

  const handleCopyCode = (e) => {
    e.stopPropagation();
    navigator.clipboard.writeText(stage.codeSnippet);
    setCopiedCode(true);
    setTimeout(() => setCopiedCode(false), 2000);
  };

  return (
    <div
      id={stage.stageId}
      className="bg-white border border-neutral-300 rounded-xl shadow-sm transition-all duration-300 overflow-hidden"
    >
      {/* Wireframe Header: </> | Code File Review Plan [Step X] */}
      <div
        onClick={onToggle}
        className="px-6 py-4 bg-neutral-50/80 border-b border-neutral-200 flex flex-wrap items-center justify-between gap-4 cursor-pointer select-none hover:bg-neutral-100/70 transition-colors"
      >
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-black text-white flex items-center justify-center font-mono font-bold text-xs shadow-sm">
            &lt;/&gt;
          </div>
          <div className="flex items-center gap-2">
            <span className="font-serif text-lg font-bold text-black tracking-tight">
              Code File Review Plan
            </span>
            <span className="text-neutral-400 font-mono">|</span>
            <span className="text-xs font-mono font-semibold text-neutral-700 bg-neutral-200/70 px-2.5 py-0.5 rounded">
              {stage.stageName}
            </span>
          </div>
        </div>

        <div className="flex items-center gap-2.5">
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-mono font-bold bg-black text-white shadow-sm">
            Step {stage.stepNumber}
          </span>
        </div>
      </div>

      {/* Wireframe Body Content - Unfolds when expanded */}
      {isExpanded && (
        <div className="p-6 space-y-6 animate-fadeIn">
          {/* Top Row: Code File Overview (Left) + File Location & Why This File Matters (Right) */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Box 1: Code File Overview */}
            <div className="lg:col-span-6 flex flex-col justify-between border border-neutral-200 rounded-lg p-5 bg-neutral-50/50 hover:border-neutral-400 transition-colors">
              <div>
                <div className="flex items-center gap-2.5 mb-2">
                  <div className="w-6 h-6 rounded bg-neutral-200 text-neutral-800 flex items-center justify-center">
                    <FileCode size={14} />
                  </div>
                  <h4 className="font-serif text-sm font-bold text-black tracking-tight">
                    Code File Overview
                  </h4>
                </div>
                <p className="text-[11px] font-mono text-neutral-500 mb-3">
                  A brief description of why this file matters (small line / few words).
                </p>
                <p className="text-xs font-sans text-neutral-800 leading-relaxed">
                  {stage.overview}
                </p>
              </div>

              <div className="mt-4 pt-3 border-t border-neutral-200 flex items-center justify-between text-[11px] font-mono text-neutral-500">
                <span>Architecture Role:</span>
                <span className="text-black font-semibold">Core Pipeline Worker</span>
              </div>
            </div>

            {/* Right Column: File Location (Box 2) + Why This File Matters (Box 3) */}
            <div className="lg:col-span-6 flex flex-col gap-4">
              {/* Box 2: File Location */}
              <div className="border border-neutral-200 rounded-lg p-4 bg-white hover:border-neutral-400 transition-colors">
                <div className="flex items-center justify-between mb-1.5">
                  <div className="flex items-center gap-2">
                    <div className="w-5 h-5 rounded bg-neutral-100 text-neutral-800 flex items-center justify-center">
                      <MapPin size={13} />
                    </div>
                    <h4 className="font-serif text-xs font-bold text-black">
                      File Location
                    </h4>
                  </div>
                  <button
                    onClick={handleCopyPath}
                    className="flex items-center gap-1 text-[10px] font-mono text-neutral-500 hover:text-black hover:bg-neutral-100 px-2 py-0.5 rounded transition-colors"
                    title="Copy relative file path"
                  >
                    {copiedPath ? (
                      <>
                        <Check size={11} className="text-emerald-600" />
                        <span className="text-emerald-600 font-semibold">Copied!</span>
                      </>
                    ) : (
                      <>
                        <Copy size={11} />
                        <span>Copy Path</span>
                      </>
                    )}
                  </button>
                </div>
                <p className="text-[10px] font-mono text-neutral-400 mb-2">
                  Where to find this file in the project.
                </p>
                <div className="font-mono text-xs font-semibold text-neutral-900 bg-neutral-100 px-3 py-2 rounded border border-neutral-200 select-all overflow-x-auto">
                  {stage.fileLocation}
                </div>
              </div>

              {/* Box 3: Why This File Matters? (3 key points) */}
              <div className="border border-neutral-200 rounded-lg p-4 bg-white flex-1 hover:border-neutral-400 transition-colors">
                <div className="flex items-center gap-2 mb-1">
                  <div className="w-5 h-5 rounded bg-amber-100 text-amber-800 flex items-center justify-center">
                    <Star size={13} className="fill-amber-500 text-amber-600" />
                  </div>
                  <h4 className="font-serif text-xs font-bold text-black">
                    Why This File Matters?
                  </h4>
                </div>
                <p className="text-[10px] font-mono text-neutral-400 mb-2.5">
                  (3 key points)
                </p>
                <ul className="space-y-2">
                  {stage.whyItMatters.map((point, pIdx) => (
                    <li key={pIdx} className="flex items-start gap-2 text-xs font-sans text-neutral-700">
                      <span className="w-4 h-4 rounded-full bg-neutral-100 text-neutral-800 font-mono text-[10px] flex items-center justify-center shrink-0 mt-0.5 border border-neutral-300">
                        {pIdx + 1}
                      </span>
                      <span className="leading-snug">{point}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>

          {/* Middle Row: Important Code Lines (Left) + Line-by-Line Explanation (Right) */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Box 4: Important Code Lines */}
            <div className="lg:col-span-6 border border-neutral-200 rounded-lg p-5 bg-neutral-900 text-neutral-100 flex flex-col justify-between shadow-inner">
              <div>
                <div className="flex items-center justify-between pb-3 mb-3 border-b border-neutral-800">
                  <div className="flex items-center gap-2">
                    <div className="w-6 h-6 rounded bg-neutral-800 text-neutral-300 flex items-center justify-center">
                      <Code2 size={14} />
                    </div>
                    <div>
                      <h4 className="font-serif text-xs font-bold text-white tracking-tight">
                        Important Code Lines
                      </h4>
                      <p className="text-[10px] font-mono text-neutral-400">
                        Select the key lines that matter.
                      </p>
                    </div>
                  </div>
                  <button
                    onClick={handleCopyCode}
                    className="flex items-center gap-1 text-[10px] font-mono text-neutral-400 hover:text-white bg-neutral-800/80 px-2 py-1 rounded transition-colors"
                  >
                    {copiedCode ? (
                      <>
                        <Check size={11} className="text-emerald-400" />
                        <span className="text-emerald-400 font-semibold">Copied!</span>
                      </>
                    ) : (
                      <>
                        <Copy size={11} />
                        <span>Copy Snippet</span>
                      </>
                    )}
                  </button>
                </div>

                <div className="mb-2 text-[11px] font-mono text-neutral-400 flex items-center gap-1.5">
                  <Terminal size={12} className="text-neutral-500" />
                  <span>{stage.importantLinesLabel}</span>
                </div>

                {/* Preformatted Code Snippet */}
                <div className="bg-neutral-950 p-3.5 rounded border border-neutral-800 overflow-x-auto text-[11px] font-mono leading-relaxed text-neutral-200 selection:bg-neutral-700 selection:text-white">
                  <pre>
                    <code>{stage.codeSnippet}</code>
                  </pre>
                </div>
              </div>

              <div className="mt-3 text-[10px] font-mono text-neutral-500 flex items-center justify-between">
                <span>Language: Python 3.11</span>
                <span>Production Checked</span>
              </div>
            </div>

            {/* Box 5: Line-by-Line Explanation */}
            <div className="lg:col-span-6 border border-neutral-200 rounded-lg p-5 bg-white flex flex-col justify-between hover:border-neutral-400 transition-colors">
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <div className="w-6 h-6 rounded bg-amber-100 text-amber-800 flex items-center justify-center">
                    <Lightbulb size={14} />
                  </div>
                  <h4 className="font-serif text-xs font-bold text-black tracking-tight">
                    Line-by-Line Explanation
                  </h4>
                </div>
                <p className="text-[10px] font-mono text-neutral-400 mb-4">
                  Explain each selected line, focusing on logic.
                </p>

                <div className="space-y-3">
                  {stage.lineByLineExplanation.map((item, lIdx) => (
                    <div
                      key={lIdx}
                      className="p-3 rounded-md bg-neutral-50 border border-neutral-200/80 hover:bg-neutral-100/60 transition-colors"
                    >
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-[10px] font-mono font-bold text-black bg-neutral-200/80 px-1.5 py-0.5 rounded">
                          {item.line}
                        </span>
                        <span className="text-xs font-serif font-bold text-neutral-900">
                          {item.title}
                        </span>
                      </div>
                      <p className="text-xs font-sans text-neutral-700 leading-relaxed">
                        {item.detail}
                      </p>
                    </div>
                  ))}
                </div>
              </div>

              <div className="mt-4 pt-3 border-t border-neutral-100 text-[10px] font-mono text-neutral-400 flex items-center gap-1.5">
                <Sparkles size={11} className="text-neutral-500" />
                <span>Deterministic Algorithmic Execution</span>
              </div>
            </div>
          </div>

          {/* Bottom Row: Simple Visual Representation (Box 6) */}
          <div className="border border-neutral-200 rounded-lg p-5 bg-white hover:border-neutral-400 transition-colors">
            <div className="flex items-center justify-between mb-1 pb-2 border-b border-neutral-100">
              <div className="flex items-center gap-2">
                <div className="w-6 h-6 rounded bg-neutral-100 text-neutral-800 flex items-center justify-center">
                  <ImageIcon size={14} />
                </div>
                <h4 className="font-serif text-xs font-bold text-black tracking-tight">
                  Simple Visual Representation
                </h4>
              </div>
              <span className="text-[10px] font-mono text-neutral-500 bg-neutral-100 px-2 py-0.5 rounded border border-neutral-200">
                Data Flow Diagram
              </span>
            </div>
            <p className="text-[10px] font-mono text-neutral-400 mb-5">
              Show what happens in this file using a small diagram.
            </p>

            {/* Clean, minimalist visual flowchart */}
            <div className="p-4 bg-neutral-50/70 border border-neutral-200/70 rounded-lg">
              <div className="grid grid-cols-1 md:grid-cols-11 gap-3 items-center text-center">
                {/* Input Node */}
                <div className="md:col-span-3 p-3.5 bg-white border border-neutral-300 rounded-lg shadow-2xs">
                  <span className="text-[9px] font-mono font-bold text-neutral-500 uppercase tracking-wider block mb-1">
                    Input Artifact
                  </span>
                  <span className="text-xs font-mono font-semibold text-neutral-900 block">
                    {stage.diagram.input}
                  </span>
                </div>

                {/* Arrow */}
                <div className="md:col-span-1 flex justify-center text-neutral-400">
                  <ArrowRight size={18} className="hidden md:block" />
                  <span className="md:hidden text-xs font-mono text-neutral-400">↓</span>
                </div>

                {/* Processing Engine Node */}
                <div className="md:col-span-3 p-3.5 bg-black text-white rounded-lg shadow-sm">
                  <span className="text-[9px] font-mono font-bold text-neutral-400 uppercase tracking-wider block mb-1">
                    Processing Transformation
                  </span>
                  <span className="text-xs font-mono font-semibold text-white block">
                    {stage.diagram.process}
                  </span>
                  {stage.diagram.fallback && (
                    <span className="text-[10px] font-mono text-amber-300 block mt-1 pt-1 border-t border-neutral-800">
                      Fallback: {stage.diagram.fallback}
                    </span>
                  )}
                </div>

                {/* Arrow */}
                <div className="md:col-span-1 flex justify-center text-neutral-400">
                  <ArrowRight size={18} className="hidden md:block" />
                  <span className="md:hidden text-xs font-mono text-neutral-400">↓</span>
                </div>

                {/* Output Node */}
                <div className="md:col-span-3 p-3.5 bg-white border border-neutral-300 rounded-lg shadow-2xs">
                  <span className="text-[9px] font-mono font-bold text-neutral-500 uppercase tracking-wider block mb-1">
                    Output Result
                  </span>
                  <span className="text-xs font-mono font-semibold text-neutral-900 block">
                    {stage.diagram.output}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
export default CodeReviewCard;

