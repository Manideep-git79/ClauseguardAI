import React, { useState, useEffect } from "react";
import { ChatTab } from "../components/document/ChatTab";
import { api } from "../api/client";
import { Bot, FileText, Sparkles, ShieldCheck } from "lucide-react";

export function StandaloneChat({ onNavigate }) {
  const [documents, setDocuments] = useState([]);
  const [selectedDocId, setSelectedDocId] = useState("");
  const [loadingDocs, setLoadingDocs] = useState(true);

  useEffect(() => {
    let isMounted = true;
    api
      .getDocuments()
      .then((docs) => {
        if (isMounted) {
          setDocuments(docs || []);
        }
      })
      .catch((err) => {
        console.warn("Could not load documents for chat picker:", err);
      })
      .finally(() => {
        if (isMounted) setLoadingDocs(false);
      });

    return () => {
      isMounted = false;
    };
  }, []);

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      <div className="border border-neutral-200 bg-white p-6 rounded-xl flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 shadow-sm">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 border border-neutral-300 bg-neutral-100 flex items-center justify-center text-black rounded-xl">
            <Bot className="w-5 h-5" />
          </div>
          <div>
            <h1 className="font-serif text-xl sm:text-2xl font-bold text-black flex items-center gap-2">
              ClauseGuard AI Chat Copilot
              <span className="text-[10px] font-mono border border-neutral-300 bg-neutral-100 text-black px-2 py-0.5 rounded font-semibold">
                Grounded RAG
              </span>
            </h1>
            <p className="text-xs font-mono text-neutral-600 mt-1">
              Ask legal copilot questions, analyze clauses, or inspect specific uploaded documents with verified citations.
            </p>
          </div>
        </div>

        <div className="w-full sm:w-auto flex flex-col sm:flex-row items-start sm:items-center gap-2">
          <label className="text-xs font-mono text-neutral-600 whitespace-nowrap flex items-center gap-1.5 font-medium">
            <FileText className="w-3.5 h-3.5 text-neutral-500" />
            Context Scope:
          </label>
          <select
            value={selectedDocId}
            onChange={(e) => setSelectedDocId(e.target.value)}
            disabled={loadingDocs}
            className="w-full sm:w-64 bg-white border border-neutral-300 text-black text-xs font-mono py-2 px-3 focus:outline-none focus:border-black rounded shadow-sm"
          >
            <option value="">General Copilot Mode</option>
            {documents.map((d) => (
              <option key={d.id} value={d.id}>
                {d.filename} ({d.document_type ? d.document_type.replace("_", " ") : "Document"})
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="border border-neutral-200 bg-white rounded-lg shadow-sm overflow-hidden">
        <ChatTab
          key={selectedDocId || "general"}
          documentId={selectedDocId || null}
        />
      </div>

      <div className="flex flex-wrap items-center justify-between text-[11px] font-mono text-neutral-500 px-2 py-1">
        <div className="flex items-center gap-2">
          <ShieldCheck className="w-3.5 h-3.5 text-neutral-600" />
          <span>Strict Hallucination Prevention • Confidence Scoring • Citation Excerpts</span>
        </div>
        <div className="flex items-center gap-2">
          <Sparkles className="w-3.5 h-3.5 text-neutral-600" />
          <span>Neural RAG + Legal Analysis Engine</span>
        </div>
      </div>
    </div>
  );
}

export default StandaloneChat;
