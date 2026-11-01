const rawBaseUrl = import.meta.env.VITE_API_URL || import.meta.env.VITE_API_BASE_URL || "";

function resolveBaseUrl() {
  if (!rawBaseUrl || rawBaseUrl.trim() === "") {
    // In dev mode, Vite proxy routes /api to the configured backend
    if (import.meta.env.DEV) {
      return "/api";
    }
    // In production without an explicit env variable, route to the live Railway backend
    return "https://clauseguard-ai-production.up.railway.app/api";
  }
  const trimmed = rawBaseUrl.trim().replace(/\/+$/, "");
  if (trimmed.endsWith("/api")) {
    return trimmed;
  }
  return `${trimmed}/api`;
}

export const BASE_URL = resolveBaseUrl();

export function apiUrl(endpoint) {
  const clean = endpoint.startsWith("/") ? endpoint : `/${endpoint}`;
  return `${BASE_URL}${clean}`;
}

async function request(endpoint, options = {}) {
  const url = apiUrl(endpoint);
  const headers = { ...(options.headers || {}) };

  try {
    const token = localStorage.getItem("cg_token") || localStorage.getItem("token");
    if (token && !headers["Authorization"]) {
      headers["Authorization"] = `Bearer ${token}`;
    }
  } catch (e) {

  }

  const fetchOptions = {
    ...options,
    credentials: "include",
    headers,
  };

  const response = await fetch(url, fetchOptions);

  if (!response.ok) {
    let errMsg = `Request failed: ${response.status} ${response.statusText}`;
    try {
      const errData = await response.json();
      if (errData && errData.error) {
        errMsg = errData.error;
      }
    } catch {

    }
    throw new Error(errMsg);
  }

  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/pdf")) {
    return response.blob();
  }

  if (contentType.includes("text/html")) {
    throw new Error(
      "Received HTML instead of JSON from API. Please verify the backend URL is active and reachable."
    );
  }

  return response.json();
}

export const api = {

  login: (email, password) =>
    request("/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    }),
  register: (payload) =>
    request("/auth/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  getMe: () => request("/auth/me"),
  logout: () => request("/auth/logout", { method: "POST" }),

  getAuditLogs: (limit = 200) => request(`/audit?limit=${limit}`),

  getDocuments: () => request("/documents"),
  getDocument: (id) => request(`/documents/${id}`),
  getDocumentStatus: (id) => request(`/documents/${id}/status`),
  uploadDocument: (formData) =>
    request("/documents", {
      method: "POST",
      body: formData,
    }),
  deleteDocument: (id) =>
    request(`/documents/${id}`, {
      method: "DELETE",
    }),
  reanalyzeDocument: (id) =>
    request(`/documents/${id}/reanalyze`, {
      method: "POST",
    }),

  getClauses: (id) => request(`/documents/${id}/clauses`),
  getMissingClauses: (id) => request(`/documents/${id}/missing-clauses`),
  getDeadlines: (id) => request(`/documents/${id}/deadlines`),
  getPiiSummary: (id) => request(`/documents/${id}/pii-summary`),
  getIntegrityFindings: (id) => request(`/documents/${id}/integrity-findings`),
  getReportPdf: (id) => request(`/documents/${id}/report.pdf`),
  getReportPdfUrl: (id) => {
    let token = "";
    try {
      token = localStorage.getItem("cg_token") || localStorage.getItem("token") || "";
    } catch {

    }
    const query = token ? `?token=${encodeURIComponent(token)}` : "";
    return `${BASE_URL}/documents/${id}/report.pdf${query}`;
  },
  downloadReportPdf: async (id, filename = "document") => {
    const blob = await api.getReportPdf(id);
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.style.display = "none";
    a.href = url;
    const cleanName = filename ? filename.replace(/\.[^/.]+$/, "") : "document";
    a.download = `clauseguard_report_${cleanName}.pdf`;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => {
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    }, 100);
  },

  sendChatMessage: (docId, question, sessionId = null) =>
    request(`/documents/${docId}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, session_id: sessionId }),
    }),
  getChatHistory: (docId) => request(`/documents/${docId}/chat/history`),

  sendStandaloneChat: (question, sessionId = null) =>
    request("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, session_id: sessionId }),
    }),
};
