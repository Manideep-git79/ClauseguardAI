import React, { useState } from "react";
import { COMPILATION_STAGES } from "../data/codeReviewPlanData";
import { CodeReviewCard } from "../components/architecture/CodeReviewCard";
import {
  Code2,
  FileCode2,
  ChevronRight,
  ChevronLeft,
  Layers,
  Sparkles,
  ArrowRight,
  Maximize2,
  Minimize2,
  CheckCircle2,
  BookOpen,
} from "lucide-react";

export function CodeReviewPlan({ onNavigate }) {
  // By default, Stage 1 (Step 1) is active
  const [activeStep, setActiveStep] = useState(1);
  const [viewMode, setViewMode] = useState("single"); // 'single' (focused step) or 'all' (all stages unfolded)

  const currentStage = COMPILATION_STAGES.find((s) => s.stepNumber === activeStep) || COMPILATION_STAGES[0];

  const handleNext = () => {
    if (activeStep < COMPILATION_STAGES.length) {
      setActiveStep(activeStep + 1);
      window.scrollTo({ top: 220, behavior: "smooth" });
    }
  };

  const handlePrev = () => {
    if (activeStep > 1) {
      setActiveStep(activeStep - 1);
      window.scrollTo({ top: 220, behavior: "smooth" });
    }
  };

  const handleSelectStep = (stepNumber) => {
    setActiveStep(stepNumber);
    if (viewMode === "all") {
      const el = document.getElementById(`stage-${stepNumber}`);
      if (el) {
        el.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    }
  };

  return (
    <div className="min-h-screen bg-neutral-50/50 pb-20 selection:bg-neutral-900 selection:text-white">
      {/* Top Header Section */}
      <div className="bg-white border-b border-neutral-200 py-8 px-4 sm:px-6 lg:px-8">
        <div className="max-w-6xl mx-auto space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <span className="px-2.5 py-0.5 rounded text-[10px] font-mono uppercase tracking-widest bg-black text-white font-semibold">
                  DocuGuard AI · Architectural Walkthrough
                </span>
                <span className="text-neutral-400 font-mono text-xs">|</span>
                <span className="text-xs font-mono text-neutral-600">
                  9 Compilation Stages
                </span>
              </div>
              <h1 className="font-serif text-2xl sm:text-3xl font-bold tracking-tight text-black">
                Code File Review Plan
              </h1>
              <p className="text-xs sm:text-sm font-sans text-neutral-600 max-w-2xl">
                A clean, step-by-step compilation walkthrough explaining how legal documents pass through
                ingestion, local privacy redaction, neural classification, deterministic risk scoring, and grounded RAG.
              </p>
            </div>

            {/* View Mode Toggle */}
            <div className="flex items-center gap-2 bg-neutral-100 p-1 rounded-lg border border-neutral-200">
              <button
                onClick={() => setViewMode("single")}
                className={`px-3 py-1.5 rounded text-xs font-mono transition-all flex items-center gap-1.5 ${
                  viewMode === "single"
                    ? "bg-white text-black font-semibold shadow-2xs"
                    : "text-neutral-600 hover:text-black"
                }`}
              >
                <Minimize2 size={13} />
                <span>Single Stage Focus</span>
              </button>
              <button
                onClick={() => setViewMode("all")}
                className={`px-3 py-1.5 rounded text-xs font-mono transition-all flex items-center gap-1.5 ${
                  viewMode === "all"
                    ? "bg-white text-black font-semibold shadow-2xs"
                    : "text-neutral-600 hover:text-black"
                }`}
              >
                <Maximize2 size={13} />
                <span>Unfold All Stages</span>
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 mt-8 space-y-8">
        {/* Stages of Compilation: Interactive Stepper Buttons */}
        <div className="bg-white border border-neutral-200 rounded-xl p-5 shadow-2xs">
          <div className="flex items-center justify-between mb-3 pb-2 border-b border-neutral-100">
            <div className="flex items-center gap-2">
              <div className="w-5 h-5 rounded bg-black text-white flex items-center justify-center font-mono text-[10px]">
                #
              </div>
              <h3 className="font-serif text-xs font-bold text-black uppercase tracking-wider">
                Stages of Compilation (Click to Unfold)
              </h3>
            </div>
            <span className="text-[10px] font-mono text-neutral-500">
              Flow: Step {activeStep} of {COMPILATION_STAGES.length}
            </span>
          </div>

          {/* Minimalist Stage Selector Buttons */}
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 lg:grid-cols-9 gap-2">
            {COMPILATION_STAGES.map((st) => {
              const isCurrent = activeStep === st.stepNumber;
              return (
                <button
                  key={st.stepNumber}
                  onClick={() => handleSelectStep(st.stepNumber)}
                  className={`text-left p-2.5 rounded-lg border transition-all flex flex-col justify-between group ${
                    isCurrent
                      ? "bg-black text-white border-black shadow-sm ring-1 ring-black"
                      : "bg-neutral-50/70 border-neutral-200 hover:border-neutral-400 hover:bg-neutral-100/60 text-neutral-800"
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span
                      className={`text-[9px] font-mono font-bold px-1.5 py-0.5 rounded ${
                        isCurrent
                          ? "bg-neutral-800 text-neutral-100"
                          : "bg-neutral-200 text-neutral-700 group-hover:bg-neutral-300"
                      }`}
                    >
                      Step {st.stepNumber}
                    </span>
                    {isCurrent && (
                      <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                    )}
                  </div>
                  <span className="text-[11px] font-serif font-bold leading-tight line-clamp-1 block">
                    {st.shortLabel}
                  </span>
                  <span
                    className={`text-[9px] font-mono truncate block mt-1 ${
                      isCurrent ? "text-neutral-400" : "text-neutral-500"
                    }`}
                  >
                    {st.fileLocation.split("/").pop()}
                  </span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Unfolded Review Plan Details View */}
        {viewMode === "single" ? (
          <div className="space-y-6">
            {/* The Active Stage Review Card (Image 2 format) */}
            <CodeReviewCard stage={currentStage} isExpanded={true} />

            {/* Stepper Navigation Controls */}
            <div className="flex items-center justify-between pt-4 pb-8 border-t border-neutral-200">
              <button
                onClick={handlePrev}
                disabled={activeStep === 1}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-mono border transition-all ${
                  activeStep === 1
                    ? "opacity-40 cursor-not-allowed border-neutral-200 text-neutral-400"
                    : "border-neutral-300 bg-white text-black hover:bg-neutral-100 shadow-2xs"
                }`}
              >
                <ChevronLeft size={14} />
                <span>Previous Stage</span>
              </button>

              <div className="text-center font-mono text-xs text-neutral-500">
                Stage {activeStep} of {COMPILATION_STAGES.length}
              </div>

              <button
                onClick={handleNext}
                disabled={activeStep === COMPILATION_STAGES.length}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-mono border transition-all ${
                  activeStep === COMPILATION_STAGES.length
                    ? "opacity-40 cursor-not-allowed border-neutral-200 text-neutral-400"
                    : "bg-black text-white border-black hover:bg-neutral-800 shadow-sm"
                }`}
              >
                <span>Next Stage</span>
                <ChevronRight size={14} />
              </button>
            </div>
          </div>
        ) : (
          /* All Stages Unfolded Mode */
          <div className="space-y-8">
            {COMPILATION_STAGES.map((stage) => (
              <div key={stage.stepNumber} className="relative">
                <CodeReviewCard stage={stage} isExpanded={true} />
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default CodeReviewPlan;

