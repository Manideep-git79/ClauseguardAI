import React from "react";
import { Clock, Calendar, AlertCircle, CheckCircle2, Calculator, HelpCircle } from "lucide-react";

export function DeadlineTimeline({ deadlines = [] }) {
  return (
    <div className="border border-neutral-200 bg-white p-6 space-y-4 shadow-sm">
      <div className="flex items-center justify-between border-b border-neutral-200 pb-3">
        <div>
          <span className="text-[11px] font-mono uppercase tracking-widest text-neutral-500 block font-semibold">
            Time-Sensitive Obligations Panel
          </span>
          <h3 className="font-serif text-lg font-bold text-black">
            Deadlines & Temporal Reasoning ({deadlines.length})
          </h3>
        </div>
        <span className="text-xs font-mono text-neutral-500 font-semibold">
          Chronological / Urgency Order
        </span>
      </div>

      <p className="text-xs text-neutral-600 leading-relaxed font-sans">
        Actionable notice periods, payment schedules, and calendar-bound deadlines extracted from contract clauses.
        Event-based relative deadlines remain unresolved until the triggering event occurs and do not feature speculative countdowns.
      </p>

      {deadlines && deadlines.length > 0 ? (
        <div className="space-y-3 pt-2">
          {deadlines.map((dl, idx) => {
            const isReview =
              dl.status === "NEEDS_REVIEW" ||
              dl.confidence === "needs_review" ||
              Boolean(dl.review_reason);
            const isCalculated = dl.status === "RESOLVED_CALCULATED";
            const isUnresolved = dl.status === "UNRESOLVED_EVENT_BASED" || (!dl.parsed_date && dl.duration_count);

            return (
              <div
                key={dl.id || idx}
                className={`border p-4 space-y-3 transition-colors ${
                  isReview
                    ? "border-amber-300 bg-amber-50/50"
                    : isCalculated
                    ? "border-sky-300 bg-sky-50/30"
                    : "border-neutral-200 bg-neutral-50/40 hover:border-neutral-400"
                }`}
              >
                {/* Header row: Type, Title, and Timeline Status Badges */}
                <div className="flex flex-wrap items-center justify-between gap-2 border-b border-neutral-200/60 pb-2">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="w-6 h-6 border border-neutral-300 bg-white flex items-center justify-center text-black text-xs shrink-0">
                      {dl.parsed_date ? (
                        isCalculated ? (
                          <Calculator className="w-3.5 h-3.5 text-sky-700" />
                        ) : (
                          <Calendar className="w-3.5 h-3.5 text-black" />
                        )
                      ) : (
                        <Clock className="w-3.5 h-3.5 text-neutral-500" />
                      )}
                    </span>
                    <span className="font-serif font-bold text-sm text-black">
                      {dl.obligation_title ||
                        (dl.deadline_type
                          ? dl.deadline_type.replace(/_/g, " ").toUpperCase()
                          : "OBLIGATION")}
                    </span>
                    {dl.page_number && (
                      <span className="text-[10px] font-mono px-1.5 py-0.5 border border-neutral-300 bg-white text-neutral-600 font-medium">
                        Page {dl.page_number}
                      </span>
                    )}
                  </div>

                  <div className="flex items-center gap-2 flex-wrap">
                    {/* Explicit or Calculated Date Badge */}
                    {dl.parsed_date && (
                      <span className={`text-xs font-mono px-2 py-0.5 border font-semibold ${
                        isCalculated
                          ? "border-sky-300 bg-sky-100 text-sky-900"
                          : "border-neutral-300 bg-white text-black"
                      }`}>
                        {dl.parsed_date}
                        {isCalculated && " (CALCULATED)"}
                      </span>
                    )}

                    {/* Relative Duration Badge */}
                    {dl.duration_count && (
                      <span className="text-xs font-mono px-2 py-0.5 border border-neutral-300 bg-white text-black font-semibold">
                        {dl.duration_count} {dl.duration_unit ? dl.duration_unit.replace(/_/g, " ").toUpperCase() : "DAYS"}
                      </span>
                    )}

                    {/* Status / Review Tag */}
                    {isReview ? (
                      <span className="text-[10px] font-mono uppercase px-2 py-0.5 border border-amber-300 bg-amber-100 text-amber-900 flex items-center gap-1 font-semibold">
                        <AlertCircle className="w-3 h-3 text-amber-700" />
                        Needs Review
                      </span>
                    ) : isCalculated ? (
                      <span className="text-[10px] font-mono uppercase px-2 py-0.5 border border-sky-300 bg-sky-100 text-sky-800 flex items-center gap-1 font-semibold">
                        <CheckCircle2 className="w-3 h-3 text-sky-600" />
                        Resolved Date
                      </span>
                    ) : isUnresolved ? (
                      <span className="text-[10px] font-mono uppercase px-2 py-0.5 border border-neutral-300 bg-white text-neutral-700 flex items-center gap-1 font-semibold">
                        <HelpCircle className="w-3 h-3 text-neutral-500" />
                        Event-Based
                      </span>
                    ) : (
                      <span className="text-[10px] font-mono uppercase px-2 py-0.5 border border-green-300 bg-green-50 text-green-800 flex items-center gap-1 font-semibold">
                        <CheckCircle2 className="w-3 h-3 text-green-600" />
                        Confident
                      </span>
                    )}
                  </div>
                </div>

                {/* Structured Obligation Metadata Fields */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs">
                  {dl.responsible_party && (
                    <div className="flex items-center gap-1.5">
                      <span className="font-mono text-neutral-500 font-medium">Party:</span>
                      <span className="font-medium text-neutral-900">{dl.responsible_party}</span>
                    </div>
                  )}

                  {dl.required_action && (
                    <div className="flex items-center gap-1.5">
                      <span className="font-mono text-neutral-500 font-medium">Action:</span>
                      <span className="font-medium text-neutral-900">{dl.required_action}</span>
                    </div>
                  )}

                  {dl.trigger_event && dl.trigger_event !== "explicit_calendar_date" && (
                    <div className="flex items-center gap-1.5">
                      <span className="font-mono text-neutral-500 font-medium">Trigger Event:</span>
                      <span className="font-medium text-neutral-900 italic">{dl.trigger_event}</span>
                    </div>
                  )}

                  {dl.recurrence && (
                    <div className="flex items-center gap-1.5">
                      <span className="font-mono text-neutral-500 font-medium">Recurrence:</span>
                      <span className="font-medium text-neutral-900">{dl.recurrence}</span>
                    </div>
                  )}

                  {dl.day_count_convention && dl.day_count_convention !== "unknown" && (
                    <div className="flex items-center gap-1.5">
                      <span className="font-mono text-neutral-500 font-medium">Convention:</span>
                      <span className="font-medium text-neutral-700">{dl.day_count_convention.replace(/_/g, " ")}</span>
                    </div>
                  )}

                  {dl.conditions && (
                    <div className="flex items-center gap-1.5 col-span-1 md:col-span-2">
                      <span className="font-mono text-neutral-500 font-medium">Conditions:</span>
                      <span className="font-medium text-neutral-800">{dl.conditions}</span>
                    </div>
                  )}
                </div>

                {/* Calculation / Ambiguity Explanation */}
                {dl.calculation_note && (
                  <div className="text-[11px] font-mono text-sky-900 bg-sky-50 border border-sky-200 px-2.5 py-1">
                    {dl.calculation_note}
                  </div>
                )}

                {dl.review_reason && (
                  <div className="text-[11px] font-mono text-amber-900 bg-amber-100/70 border border-amber-300 px-2.5 py-1 flex items-center gap-1.5">
                    <AlertCircle className="w-3.5 h-3.5 text-amber-700 shrink-0" />
                    <span>Review Reason: {dl.review_reason}</span>
                  </div>
                )}

                {/* Supporting Clause Snippet in Privacy-Safe Text */}
                <p className="text-xs text-neutral-700 font-sans leading-relaxed border-t border-neutral-200/50 pt-2 italic">
                  "{dl.raw_text}"
                </p>
              </div>
            );
          })}
        </div>
      ) : (
        <div className="border border-neutral-200 bg-white p-6 text-center text-xs text-neutral-500 font-mono shadow-sm">
          No calendar deadlines or relative notice periods detected in this document.
        </div>
      )}
    </div>
  );
}
