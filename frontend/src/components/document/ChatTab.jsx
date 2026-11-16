import React, { useState, useEffect, useRef } from "react";
import { LegalMarkdown } from "../chat/LegalMarkdown";
import { apiUrl } from "../../api/client";

export const ChatTab = ({ documentId }) => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(true);
  const [sending, setSending] = useState(false);
  const chatWindowRef = useRef(null);

  const suggestions = [
    "What are the termination conditions?",
    "Is there an automatic renewal clause?",
    "What is the required notice period?",
    "Summarize the main liabilities & risks",
  ];

  useEffect(() => {
    let isMounted = true;
    const fetchHistory = async () => {
      try {
        const path = documentId
          ? `/chat/history?documentId=${documentId}`
          : "/chat/history";
        const token = localStorage.getItem("cg_token") || localStorage.getItem("token");
        const headers = {};
        if (token) headers["Authorization"] = `Bearer ${token}`;

        const res = await fetch(apiUrl(path), {
          credentials: "include",
          headers,
        });
        if (res.ok) {
          const data = await res.json();
          if (isMounted) {
            setMessages(data.messages || []);
          }
        }
      } catch (err) {
        console.error("Failed to load chat history:", err);
      } finally {
        if (isMounted) setLoading(false);
      }
    };
    fetchHistory();
    return () => {
      isMounted = false;
    };
  }, [documentId]);

  useEffect(() => {
    if (chatWindowRef.current) {
      chatWindowRef.current.scrollTop = chatWindowRef.current.scrollHeight;
    }
  }, [messages, sending]);

  const handleSend = async (customText) => {
    const q = (customText || input).trim();
    if (!q || sending) return;

    const userMsg = {
      id: `temp-${Date.now()}`,
      role: "USER",
      content: q,
      created_at: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, userMsg]);
    if (!customText) setInput("");
    setSending(true);

    try {
      const token = localStorage.getItem("cg_token") || localStorage.getItem("token");
      const headers = { "Content-Type": "application/json" };
      if (token) headers["Authorization"] = `Bearer ${token}`;

      const res = await fetch(apiUrl("/chat"), {
        method: "POST",
        headers,
        credentials: "include",
        body: JSON.stringify({ question: q, documentId }),
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.error || "Unable to generate a response right now. Please try again.");
      }

      const data = await res.json();
      const assistantMsg = {
        id: data.id || `assistant-${Date.now()}`,
        role: "ASSISTANT",
        content: data.answer,
        confidence: data.confidence,
        grounded: data.grounded,
        sources: data.sources || [],
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: `err-${Date.now()}`,
          role: "ASSISTANT",
          content: err.message || "Unable to generate a response right now. Please try again.",
          grounded: false,
        },
      ]);
    } finally {
      setSending(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  if (loading) {
    return (
      <div style={{ padding: "32px", textAlign: "center", color: "#666", background: "#ffffff" }}>
        Loading AI Chat Session…
      </div>
    );
  }

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        height: "650px",
        border: "1px solid #e5e5e5",
        borderRadius: "12px",
        background: "#ffffff",
        color: "#000000",
        overflow: "hidden",
      }}
    >
      <div
        style={{
          padding: "16px 20px",
          borderBottom: "1px solid #e5e5e5",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          background: "#fafafa",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "8px", fontWeight: 600, fontSize: "15px", color: "#000000" }}>
          <span style={{ height: "8px", width: "8px", borderRadius: "50%", background: "#000000" }} />
          AI Legal Copilot
        </div>
        <span
          style={{
            fontSize: "11px",
            background: "#f4f4f5",
            padding: "4px 8px",
            borderRadius: "4px",
            color: "#18181b",
            border: "1px solid #e4e4e7",
            fontWeight: 500,
          }}
        >
          Grounded Copilot
        </span>
      </div>

      <div
        style={{
          padding: "12px 16px",
          display: "flex",
          gap: "8px",
          overflowX: "auto",
          borderBottom: "1px solid #e5e5e5",
          background: "#fafafa",
        }}
      >
        {suggestions.map((s, idx) => (
          <button
            key={idx}
            onClick={() => handleSend(s)}
            disabled={sending}
            style={{
              whiteSpace: "nowrap",
              padding: "6px 14px",
              borderRadius: "20px",
              border: "1px solid #d4d4d8",
              background: "#ffffff",
              color: "#18181b",
              fontSize: "12px",
              cursor: sending ? "not-allowed" : "pointer",
              fontWeight: 500,
              boxShadow: "0 1px 2px rgba(0,0,0,0.03)",
            }}
          >
            {s}
          </button>
        ))}
      </div>

      <div
        ref={chatWindowRef}
        style={{
          flex: 1,
          padding: "20px",
          overflowY: "auto",
          display: "flex",
          flexDirection: "column",
          gap: "16px",
          background: "#fcfcfc",
        }}
      >
        {messages.length === 0 ? (
          <div style={{ margin: "auto", textAlign: "center", color: "#71717a" }}>
            <div style={{ fontSize: "32px", marginBottom: "8px" }}>⚖️</div>
            <div style={{ fontWeight: 600, color: "#000000", fontSize: "16px" }}>How can I assist you today?</div>
            <div style={{ fontSize: "13px", color: "#71717a", marginTop: "4px" }}>Ask any question or click a suggestion chip above.</div>
          </div>
        ) : (
          messages.map((m, idx) => {
            const isUser = m.role === "USER" || m.role === "user";
            return (
              <div
                key={m.id || idx}
                style={{
                  alignSelf: isUser ? "flex-end" : "flex-start",
                  maxWidth: "85%",
                  background: isUser ? "#000000" : "#ffffff",
                  color: isUser ? "#ffffff" : "#000000",
                  border: isUser ? "1px solid #000000" : "1px solid #e5e5e5",
                  borderRadius: isUser ? "16px 16px 2px 16px" : "16px 16px 16px 2px",
                  padding: "14px 18px",
                  fontSize: "14px",
                  lineHeight: "1.6",
                  boxShadow: isUser ? "0 2px 6px rgba(0,0,0,0.15)" : "0 2px 8px rgba(0,0,0,0.05)",
                }}
              >
                {!isUser && (
                  <div style={{ marginBottom: "8px", display: "flex", alignItems: "center", gap: "6px" }}>
                    <span
                      style={{
                        fontSize: "10px",
                        textTransform: "uppercase",
                        padding: "2px 8px",
                        borderRadius: "4px",
                        background: "#f4f4f5",
                        color: "#18181b",
                        border: "1px solid #e4e4e7",
                        fontWeight: 600,
                      }}
                    >
                      {m.grounded ? "✓ Grounded" : "Legal Guidance"}
                    </span>
                    {typeof m.confidence === "number" && (
                      <span style={{ fontSize: "11px", color: "#71717a" }}>
                        ({Math.round(m.confidence * 100)}% confidence)
                      </span>
                    )}
                  </div>
                )}

                {isUser ? (
                  <div style={{ color: "#ffffff", whiteSpace: "pre-wrap" }}>{m.content}</div>
                ) : (
                  <LegalMarkdown content={m.content} />
                )}

                {!isUser && m.sources && m.sources.length > 0 && (
                  <div style={{ marginTop: "12px", paddingTop: "10px", borderTop: "1px solid #f0f0f0" }}>
                    <div style={{ fontSize: "11px", fontWeight: 700, color: "#52525b", marginBottom: "6px" }}>
                      📎 Citations:
                    </div>
                    {m.sources.map((src, sIdx) => (
                      <div
                        key={sIdx}
                        style={{
                          fontSize: "12px",
                          background: "#f9fafb",
                          padding: "8px 10px",
                          borderRadius: "4px",
                          fontStyle: "italic",
                          color: "#3f3f46",
                          border: "1px solid #e5e5e5",
                          marginTop: sIdx > 0 ? "6px" : "0",
                        }}
                      >
                        "{src.excerpt}"
                      </div>
                    ))}
                  </div>
                )}
              </div>
            );
          })
        )}
        {sending && (
          <div
            style={{
              alignSelf: "flex-start",
              background: "#ffffff",
              border: "1px solid #e5e5e5",
              padding: "10px 16px",
              borderRadius: "16px",
              fontSize: "13px",
              color: "#52525b",
              boxShadow: "0 2px 6px rgba(0,0,0,0.04)",
            }}
          >
            ✦ Generating clean legal response…
          </div>
        )}
      </div>

      <div
        style={{
          padding: "16px",
          borderTop: "1px solid #e5e5e5",
          background: "#fafafa",
          display: "flex",
          gap: "10px",
        }}
      >
        <input
          type="text"
          placeholder="Ask a question about this document…"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={sending}
          style={{
            flex: 1,
            padding: "12px 16px",
            borderRadius: "8px",
            border: "1px solid #d4d4d8",
            background: "#ffffff",
            color: "#000000",
            fontSize: "14px",
            outline: "none",
          }}
        />
        <button
          onClick={() => handleSend()}
          disabled={sending || !input.trim()}
          style={{
            padding: "12px 24px",
            borderRadius: "8px",
            border: "1px solid #000000",
            background: sending || !input.trim() ? "#e4e4e7" : "#000000",
            color: sending || !input.trim() ? "#a1a1aa" : "#ffffff",
            fontWeight: 600,
            cursor: sending || !input.trim() ? "not-allowed" : "pointer",
            transition: "all 0.15s ease",
          }}
        >
          Send
        </button>
      </div>
    </div>
  );
};

export default ChatTab;
