import React, { useState, useEffect } from "react";
import {
  Activity,
  FileText,
  Bot,
  ShieldCheck,
  RefreshCw,
  Search,
  CheckCircle,
  AlertCircle,
  LogIn,
  LogOut,
  UploadCloud,
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import { useAuth } from "../context/AuthContext";
import { api } from "../api/client";

export function AuditTrail({ onNavigate }) {
  const { user } = useAuth();
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filterType, setFilterType] = useState("all");
  const [searchTerm, setSearchTerm] = useState("");
  const [expandedLogId, setExpandedLogId] = useState(null);

  const fetchLogs = async () => {
    setLoading(true);
    try {
      const data = await api.getAuditLogs(200);
      setLogs(data.logs || []);
    } catch (err) {
      console.error("Failed to load audit logs:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchLogs();
  }, []);

  const filteredLogs = logs.filter((log) => {
    if (filterType === "documents" && !log.action.startsWith("DOC_")) return false;
    if (filterType === "chat" && !log.action.startsWith("CHAT_")) return false;
    if (
      filterType === "auth" &&
      !log.action.startsWith("USER_") &&
      !log.action.startsWith("LOGIN_")
    )
      return false;

    if (searchTerm.trim()) {
      const term = searchTerm.toLowerCase();
      const matchAction = log.action.toLowerCase().includes(term);
      const matchUser = (log.user_email || "").toLowerCase().includes(term);
      const matchDetails = JSON.stringify(log.details || {}).toLowerCase().includes(term);
      return matchAction || matchUser || matchDetails;
    }
    return true;
  });

  const totalEvents = logs.length;
  const docEvents = logs.filter((l) => l.action.startsWith("DOC_")).length;
  const chatEvents = logs.filter((l) => l.action.startsWith("CHAT_")).length;
  const authEvents = logs.filter(
    (l) => l.action.startsWith("USER_") || l.action.startsWith("LOGIN_")
  ).length;

  const getActionBadge = (action) => {
    switch (action) {
      case "USER_LOGIN":
        return {
          bg: "bg-neutral-100 text-black border-neutral-300",
          icon: LogIn,
          label: "User Sign In",
        };
      case "USER_REGISTER":
        return {
          bg: "bg-neutral-100 text-black border-neutral-300",
          icon: ShieldCheck,
          label: "Account Created",
        };
      case "USER_LOGOUT":
        return {
          bg: "bg-neutral-100 text-neutral-600 border-neutral-300",
          icon: LogOut,
          label: "User Sign Out",
        };
      case "LOGIN_FAILED":
        return {
          bg: "bg-red-50 text-red-700 border-red-300",
          icon: AlertCircle,
          label: "Failed Login",
        };
      case "DOC_UPLOADED":
        return {
          bg: "bg-neutral-100 text-black border-neutral-300",
          icon: UploadCloud,
          label: "Document Uploaded",
        };
      case "DOC_ANALYZED":
        return {
          bg: "bg-neutral-100 text-black border-neutral-300",
          icon: CheckCircle,
          label: "Analysis Finished",
        };
      case "DOC_DELETED":
        return {
          bg: "bg-neutral-100 text-neutral-600 border-neutral-300",
          icon: AlertCircle,
          label: "Document Deleted",
        };
      case "CHAT_QUERY":
        return {
          bg: "bg-neutral-100 text-black border-neutral-300",
          icon: Bot,
          label: "AI Chat Query",
        };
      case "CHAT_OPENED":
        return {
          bg: "bg-neutral-100 text-neutral-600 border-neutral-300",
          icon: Bot,
          label: "Chat Opened",
        };
      default:
        return {
          bg: "bg-neutral-100 text-neutral-700 border-neutral-300",
          icon: Activity,
          label: action,
        };
    }
  };

  return (
    <div className="space-y-6">
      <div className="border border-neutral-200 bg-white p-6 rounded-xl flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 shadow-sm">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-neutral-100 border border-neutral-300 flex items-center justify-center text-black shrink-0">
            <Activity className="w-5 h-5" />
          </div>
          <div>
            <h1 className="font-serif text-2xl font-bold text-black flex items-center gap-2">
              Audit & Compliance Trail
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-neutral-100 text-neutral-800 border border-neutral-300 font-semibold">
                Ledger
              </span>
            </h1>
            <p className="text-xs font-mono text-neutral-600 mt-1">
              Immutable chronological record of document uploads, automated risk evaluations, AI inquiries, and security sessions.
            </p>
          </div>
        </div>

        <button
          onClick={fetchLogs}
          disabled={loading}
          className="px-3.5 py-2 border border-neutral-300 bg-white hover:bg-neutral-100 text-black rounded text-xs font-mono flex items-center gap-2 transition-colors font-medium shadow-sm"
        >
          <RefreshCw size={14} className={loading ? "animate-spin" : ""} />
          <span>Refresh Trail</span>
        </button>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div className="border border-neutral-200 bg-white p-4 rounded-lg flex flex-col justify-between shadow-sm">
          <span className="text-[10px] font-mono text-neutral-500 uppercase tracking-wider font-semibold">
            Total Logged Events
          </span>
          <span className="font-serif text-2xl font-bold text-black mt-1">
            {totalEvents}
          </span>
          <span className="text-[10px] font-mono text-neutral-500 mt-1">User-specific actions</span>
        </div>

        <div className="border border-neutral-200 bg-white p-4 rounded-lg flex flex-col justify-between shadow-sm">
          <span className="text-[10px] font-mono text-neutral-500 uppercase tracking-wider font-semibold">
            Document Ingestions
          </span>
          <span className="font-serif text-2xl font-bold text-black mt-1">
            {docEvents}
          </span>
          <span className="text-[10px] font-mono text-neutral-500 mt-1">Uploads & Audits</span>
        </div>

        <div className="border border-neutral-200 bg-white p-4 rounded-lg flex flex-col justify-between shadow-sm">
          <span className="text-[10px] font-mono text-neutral-500 uppercase tracking-wider font-semibold">
            AI Inquiries
          </span>
          <span className="font-serif text-2xl font-bold text-black mt-1">
            {chatEvents}
          </span>
          <span className="text-[10px] font-mono text-neutral-500 mt-1">Grounded RAG Queries</span>
        </div>

        <div className="border border-neutral-200 bg-white p-4 rounded-lg flex flex-col justify-between shadow-sm">
          <span className="text-[10px] font-mono text-neutral-500 uppercase tracking-wider font-semibold">
            Auth & Sessions
          </span>
          <span className="font-serif text-2xl font-bold text-black mt-1">
            {authEvents}
          </span>
          <span className="text-[10px] font-mono text-neutral-500 mt-1">Logins & Registrations</span>
        </div>
      </div>

      <div className="border border-neutral-200 bg-white p-4 rounded-lg flex flex-col sm:flex-row items-center justify-between gap-4 shadow-sm">
        <div className="flex items-center gap-1.5 overflow-x-auto w-full sm:w-auto">
          {[
            { id: "all", label: "All Events" },
            { id: "documents", label: "Documents" },
            { id: "chat", label: "AI Inquiries" },
            { id: "auth", label: "Security & Auth" },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setFilterType(tab.id)}
              className={`px-3 py-1.5 rounded text-xs font-mono whitespace-nowrap transition-colors border ${
                filterType === tab.id
                  ? "bg-black text-white font-semibold border-black shadow-sm"
                  : "text-neutral-600 hover:text-black border-transparent hover:bg-neutral-100"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        <div className="relative w-full sm:w-64">
          <Search size={14} className="absolute left-3 top-2.5 text-neutral-400" />
          <input
            type="text"
            placeholder="Search action or details…"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full bg-neutral-50 border border-neutral-300 rounded text-xs font-mono pl-8 pr-3 py-1.5 text-black placeholder-neutral-400 focus:outline-none focus:border-black"
          />
        </div>
      </div>

      <div className="border border-neutral-200 bg-white rounded-xl overflow-hidden shadow-sm">
        {loading ? (
          <div className="p-12 text-center text-xs font-mono text-neutral-500 animate-pulse">
            Loading audit records from database…
          </div>
        ) : filteredLogs.length === 0 ? (
          <div className="p-12 text-center space-y-2">
            <Activity className="w-8 h-8 text-neutral-400 mx-auto" />
            <h4 className="font-serif text-sm text-neutral-700 font-semibold">No Audit Events Found</h4>
            <p className="text-xs font-mono text-neutral-500">
              No actions match the selected filter criteria.
            </p>
          </div>
        ) : (
          <div className="divide-y divide-neutral-200">
            {filteredLogs.map((log) => {
              const badge = getActionBadge(log.action);
              const Icon = badge.icon;
              const isExpanded = expandedLogId === log.id;
              const formattedDate = log.created_at
                ? new Date(log.created_at).toLocaleString()
                : "Just now";

              return (
                <div
                  key={log.id}
                  className="p-4 hover:bg-neutral-50 transition-colors"
                >
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                    <div className="flex items-start sm:items-center gap-3">
                      <div
                        className={`w-7 h-7 rounded border flex items-center justify-center shrink-0 ${badge.bg}`}
                      >
                        <Icon size={14} />
                      </div>
                      <div>
                        <div className="flex items-center gap-2 flex-wrap">
                          <span
                            className={`text-[10px] font-mono font-bold uppercase px-2 py-0.5 rounded border ${badge.bg}`}
                          >
                            {badge.label}
                          </span>
                          {log.resource_type && (
                            <span className="text-[10px] font-mono text-neutral-500">
                              • {log.resource_type}
                            </span>
                          )}
                          {log.user_email && (
                            <span className="text-[10px] font-mono text-neutral-700 bg-neutral-100 border border-neutral-200 px-2 py-0.5 rounded">
                              {log.user_email}
                            </span>
                          )}
                        </div>

                        <div className="text-xs font-mono text-neutral-700 mt-1">
                          {log.action === "DOC_UPLOADED" && (
                            <span>
                              Uploaded <strong>{log.details?.filename}</strong> (
                              {log.details?.document_type})
                            </span>
                          )}
                          {log.action === "DOC_ANALYZED" && (
                            <span>
                              Analysis completed for <strong>{log.details?.filename}</strong> —
                              Risk Score:{" "}
                              <strong className="text-black">
                                {log.details?.overall_risk_score}
                              </strong>{" "}
                              ({log.details?.risk_band})
                            </span>
                          )}
                          {log.action === "CHAT_QUERY" && (
                            <span>
                              Asked: <em>"{log.details?.question}"</em>
                            </span>
                          )}
                          {log.action === "USER_LOGIN" && (
                            <span>
                              Signed in from IP: <code>{log.ip_address || "127.0.0.1"}</code>
                            </span>
                          )}
                          {log.action === "USER_REGISTER" && (
                            <span>Registered account for {log.details?.name}</span>
                          )}
                          {log.action === "DOC_DELETED" && (
                            <span>Deleted document {log.details?.filename}</span>
                          )}
                          {!["DOC_UPLOADED", "DOC_ANALYZED", "CHAT_QUERY", "USER_LOGIN", "USER_REGISTER", "DOC_DELETED"].includes(log.action) && (
                            <span>{JSON.stringify(log.details || {})}</span>
                          )}
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center gap-4 self-end sm:self-center">
                      <span className="text-[11px] font-mono text-neutral-500 whitespace-nowrap">
                        {formattedDate}
                      </span>
                      <button
                        onClick={() =>
                          setExpandedLogId(isExpanded ? null : log.id)
                        }
                        className="text-neutral-500 hover:text-black p-1 rounded transition-colors"
                        title="Inspect Event JSON"
                      >
                        {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                      </button>
                    </div>
                  </div>

                  {isExpanded && (
                    <div className="mt-3 p-3 rounded bg-neutral-50 border border-neutral-200 text-[11px] font-mono space-y-1">
                      <div className="text-neutral-500 text-[10px] uppercase font-semibold">Event Payload Metadata:</div>
                      <pre className="text-neutral-800 overflow-x-auto p-2 bg-white border border-neutral-300 rounded">
                        {JSON.stringify(log, null, 2)}
                      </pre>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

export default AuditTrail;
