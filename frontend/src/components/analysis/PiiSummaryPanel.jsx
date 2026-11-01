import React, { useEffect, useState } from "react";
import { api } from "../../api/client";
import { ShieldCheck, Lock, EyeOff, AlertCircle } from "lucide-react";

export function PiiSummaryPanel({ documentId }) {
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!documentId) return;
    setLoading(true);
    api
      .getPiiSummary(documentId)
      .then((data) => {
        setSummary(data);
        setError(null);
      })
      .catch((err) => {
        setError(err.message);
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

  if (error || !summary) {
    return null;
  }

  const entityCounts = summary.entity_counts || {};
  const totalFindings = summary.total_findings || 0;
  const mappingTable = summary.mapping_table || [];

  return (
    <div className="border border-neutral-200 bg-white p-5 shadow-sm space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-neutral-200 pb-3">
        <div className="flex items-center gap-2">
          <ShieldCheck className="w-5 h-5 text-green-600" />
          <h3 className="font-serif font-bold text-lg text-black">
            PII Redaction & Pseudonymization
          </h3>
        </div>
        <div className="flex items-center gap-2 text-xs font-mono text-neutral-500">
          <Lock className="w-3.5 h-3.5 text-neutral-400" />
          <span>Encrypted at rest • Zero raw leakage</span>
        </div>
      </div>

      <p className="text-xs text-neutral-600 leading-relaxed font-sans">
        Microsoft Presidio redacts personal identifying information before clause segmentation, model inference, or storage.
        Downstream models and our AI RAG engine only ever process neutral, sequential pseudonyms.
      </p>

      {totalFindings === 0 ? (
        <div className="border border-neutral-200 bg-neutral-50 p-4 text-center">
          <p className="text-xs font-mono text-neutral-600">
            No personal identifying information entities were detected in this document.
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
            {Object.entries(entityCounts).map(([entityType, count]) => (
              <div
                key={entityType}
                className="border border-neutral-200 bg-neutral-50 p-3 flex flex-col justify-between"
              >
                <span className="text-[10px] font-mono text-neutral-500 uppercase tracking-wider block font-semibold">
                  {entityType.replace(/_/g, " ")}
                </span>
                <span className="font-serif text-xl font-bold text-black mt-1">
                  {count} <span className="text-xs font-mono text-neutral-500 font-normal">redacted</span>
                </span>
              </div>
            ))}
          </div>

          {/* Privacy-First Entity Mapping Table */}
          {mappingTable.length > 0 && (
            <div className="border border-neutral-200 bg-white overflow-hidden shadow-sm">
              <div className="bg-neutral-50 px-4 py-3 border-b border-neutral-200 flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <span className="font-serif font-bold text-sm text-black">
                    Privacy-First Pseudonym Mapping
                  </span>
                  <span className="text-[10px] font-mono text-neutral-600 bg-neutral-200 px-2 py-0.5 font-semibold rounded">
                    {mappingTable.length} Resolved {mappingTable.length === 1 ? "Entity" : "Entities"}
                  </span>
                </div>
                <span className="text-[11px] font-mono text-neutral-500">
                  Neutral sequential labels • Document-scoped
                </span>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs font-sans border-collapse">
                  <thead>
                    <tr className="border-b border-neutral-200 bg-neutral-100 font-mono text-[10px] text-neutral-600 uppercase tracking-wider">
                      <th className="px-4 py-2.5 font-bold">Replacement Label</th>
                      <th className="px-4 py-2.5 font-bold">Entity Type</th>
                      <th className="px-4 py-2.5 font-bold">Occurrences</th>
                      <th className="px-4 py-2.5 font-bold">Clause / Section References</th>
                      <th className="px-4 py-2.5 font-bold">Review Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-neutral-200 font-sans">
                    {mappingTable.map((item, idx) => (
                      <tr key={idx} className="hover:bg-neutral-50 transition-colors">
                        <td className="px-4 py-3 whitespace-nowrap">
                          <span className="font-mono font-bold text-xs bg-neutral-100 border border-neutral-300 text-black px-2 py-1 rounded">
                            {item.replacement_label}
                          </span>
                        </td>
                        <td className="px-4 py-3 font-mono text-[11px] text-neutral-700 font-semibold whitespace-nowrap">
                          {item.entity_type.replace(/_/g, " ")}
                        </td>
                        <td className="px-4 py-3 whitespace-nowrap">
                          <span className="font-mono text-xs text-neutral-800">
                            {item.occurrence_count} {item.occurrence_count === 1 ? "mention" : "mentions"}
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <div className="flex flex-wrap gap-1">
                            {item.clause_references && item.clause_references.length > 0 ? (
                              item.clause_references.map((ref, rIdx) => (
                                <span
                                  key={rIdx}
                                  className="text-[10px] font-mono text-neutral-600 bg-neutral-100 border border-neutral-200 px-1.5 py-0.5 rounded"
                                >
                                  {ref}
                                </span>
                              ))
                            ) : (
                              <span className="text-[10px] font-mono text-neutral-400">General text</span>
                            )}
                          </div>
                        </td>
                        <td className="px-4 py-3 whitespace-nowrap">
                          {item.review_status === "Needs Review" ? (
                            <span className="inline-flex items-center gap-1 text-[11px] font-mono text-amber-800 bg-amber-50 border border-amber-300 px-2 py-0.5 font-semibold rounded">
                              <AlertCircle className="w-3 h-3 text-amber-600" />
                              Needs Review
                            </span>
                          ) : (
                            <span className="inline-flex items-center gap-1 text-[11px] font-mono text-green-800 bg-green-50 border border-green-300 px-2 py-0.5 font-semibold rounded">
                              <ShieldCheck className="w-3 h-3 text-green-600" />
                              Auto-redacted
                            </span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          <div className="border border-neutral-300 bg-neutral-100 p-3 flex items-start gap-2.5 rounded">
            <EyeOff className="w-4 h-4 text-black shrink-0 mt-0.5" />
            <p className="text-xs text-neutral-800 leading-relaxed font-sans">
              <strong>Zero-Storage Guarantee:</strong> The database only records entity types and character offsets in redacted text.
              Actual names, phone numbers, or account numbers are never retained in plaintext or exposed through the API.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
