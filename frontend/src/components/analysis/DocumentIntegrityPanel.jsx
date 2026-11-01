import React, { useEffect, useState } from "react";
import { api } from "../../api/client";
import { ShieldAlert, CheckCircle2, AlertTriangle, Eye, Terminal, Info } from "lucide-react";

export function DocumentIntegrityPanel({ documentId }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!documentId) return;
    setLoading(true);
    api
      .getIntegrityFindings(documentId)
      .then((res) => {
        setData(res);
        setError(null);
      })
      .catch((err) => {
        setError(err.message || "Failed to load integrity signals.");
      })
      .finally(() => setLoading(false));
  }, [documentId]);

  if (loading) {
    return (
      <div className="border border-neutral-200 bg-white p-5 shadow-sm">
        <div className="animate-pulse space-y-3">
          <div className="h-4 bg-neutral-100 w-1/3"></div>
          <div className="h-10 bg-neutral-50 w-full"></div>
        </div>
      </div>
    );
  }

  if (error || !data) {
    return null;
  }

  const findings = data.findings || [];
  const hasSignals = findings.length > 0;
  const criticalCount = findings.filter((f) => f.severity === "critical").length;
  const highCount = findings.filter((f) => f.severity === "high").length;
  const mediumCount = findings.filter((f) => f.severity === "medium").length;

  const getSeverityBadge = (sev) => {
    switch (sev) {
      case "critical":
        return "bg-red-100 text-red-800 border-red-300";
      case "high":
        return "bg-orange-100 text-orange-800 border-orange-300";
      case "medium":
        return "bg-amber-100 text-amber-800 border-amber-300";
      default:
        return "bg-neutral-100 text-neutral-700 border-neutral-300";
    }
  };

  const formatCategory = (cat) => {
    if (!cat) return "Signal";
    return cat.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
  };

  return (
    <div className="border border-neutral-200 bg-white p-5 shadow-sm space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-neutral-200 pb-3">
        <div className="flex items-center gap-2">
          {hasSignals ? (
            <ShieldAlert className="w-5 h-5 text-amber-600" />
          ) : (
            <CheckCircle2 className="w-5 h-5 text-green-600" />
          )}
          <div>
            <h3 className="font-serif font-bold text-lg text-black">
              Document Integrity & AI Safety Signals
            </h3>
            <p className="text-xs text-neutral-500 font-mono">
              Independent document verification: hidden text, visual anomalies, and prompt injection scans
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {hasSignals ? (
            <span className="text-xs font-mono px-2.5 py-1 bg-amber-50 border border-amber-300 text-amber-800 font-bold uppercase">
              {findings.length} Signal{findings.length > 1 ? "s" : ""} Flagged
            </span>
          ) : (
            <span className="text-xs font-mono px-2.5 py-1 bg-green-50 border border-green-300 text-green-800 font-bold uppercase">
              Verified Clean (0 Signals)
            </span>
          )}
        </div>
      </div>

      <div className="bg-neutral-50 border border-neutral-200 p-3 text-xs text-neutral-600 font-mono flex items-start gap-2.5">
        <Info className="w-4 h-4 text-neutral-500 shrink-0 mt-0.5" />
        <div>
          <span>
            <strong>Deterministic Isolation Policy:</strong> Document integrity signals are tracked independently
            from legal clause risk. Formatting anomalies and prompt injection detections do not artificially inflate
            contractual risk scores, but are surfaced for manual compliance review.
          </span>
        </div>
      </div>

      {!hasSignals ? (
        <div className="py-6 text-center space-y-2">
          <CheckCircle2 className="w-8 h-8 text-green-600 mx-auto" />
          <h4 className="font-medium text-sm text-neutral-900">No Hidden Content or Injection Anomalies</h4>
          <p className="text-xs text-neutral-500 font-mono max-w-lg mx-auto">
            All text spans align with visible page coordinates, normal typography scales, and neutral contract phrasing.
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          <div className="grid grid-cols-3 gap-3 text-center">
            <div className="p-2 border border-neutral-200 bg-neutral-50">
              <span className="text-[10px] uppercase font-mono text-neutral-500 block">Critical</span>
              <span className="font-serif text-xl font-bold text-red-600">{criticalCount}</span>
            </div>
            <div className="p-2 border border-neutral-200 bg-neutral-50">
              <span className="text-[10px] uppercase font-mono text-neutral-500 block">High</span>
              <span className="font-serif text-xl font-bold text-orange-600">{highCount}</span>
            </div>
            <div className="p-2 border border-neutral-200 bg-neutral-50">
              <span className="text-[10px] uppercase font-mono text-neutral-500 block">Medium / Warning</span>
              <span className="font-serif text-xl font-bold text-amber-600">{mediumCount}</span>
            </div>
          </div>

          <div className="overflow-x-auto border border-neutral-200">
            <table className="w-full text-left text-xs font-mono">
              <thead className="bg-neutral-100 text-neutral-700 uppercase border-b border-neutral-200 text-[10px]">
                <tr>
                  <th className="p-2.5">Category</th>
                  <th className="p-2.5">Severity</th>
                  <th className="p-2.5">Page</th>
                  <th className="p-2.5">Detection Method</th>
                  <th className="p-2.5">Explanation & Evidence</th>
                  <th className="p-2.5">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-neutral-200 bg-white">
                {findings.map((f, i) => (
                  <tr key={f.id || i} className="hover:bg-neutral-50">
                    <td className="p-2.5 font-bold text-neutral-900 whitespace-nowrap">
                      {formatCategory(f.category)}
                    </td>
                    <td className="p-2.5">
                      <span className={`px-2 py-0.5 border text-[10px] uppercase font-bold rounded-sm ${getSeverityBadge(f.severity)}`}>
                        {f.severity}
                      </span>
                    </td>
                    <td className="p-2.5 text-neutral-600">
                      {f.page_number ? `P. ${f.page_number}` : "Doc"}
                    </td>
                    <td className="p-2.5 text-neutral-600">
                      {f.detection_method || "Heuristic"}
                    </td>
                    <td className="p-2.5 text-neutral-800 max-w-md">
                      <div>{f.explanation}</div>
                      {f.evidence_snippet && (
                        <div className="mt-1 text-[11px] text-neutral-500 bg-neutral-50 p-1.5 border border-neutral-200 rounded font-sans italic">
                          "{f.evidence_snippet}"
                        </div>
                      )}
                    </td>
                    <td className="p-2.5 text-neutral-600 whitespace-nowrap">
                      <span className="px-1.5 py-0.5 bg-neutral-100 text-neutral-700 border border-neutral-300 text-[10px] rounded">
                        {f.review_status || "Flagged"}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
