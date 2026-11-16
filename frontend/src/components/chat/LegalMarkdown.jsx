import React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

function sanitizeUrl(url) {
  if (!url) return "#";
  const trimmed = url.trim();
  if (
    trimmed.startsWith("http://") ||
    trimmed.startsWith("https://") ||
    trimmed.startsWith("mailto:") ||
    trimmed.startsWith("#")
  ) {
    return trimmed;
  }
  return "#";
}

function renderRiskBadge(text) {
  if (typeof text !== "string") return text;

  const riskPattern = /\b(CRITICAL|HIGH|MEDIUM|LOW)\s+(RISK|LEVEL)\b|\bRISK\s*:\s*(CRITICAL|HIGH|MEDIUM|LOW)\b/gi;
  if (!riskPattern.test(text)) return text;

  const parts = [];
  let lastIndex = 0;
  let match;
  const regex = /\b(CRITICAL|HIGH|MEDIUM|LOW)\s+(RISK|LEVEL)\b|\bRISK\s*:\s*(CRITICAL|HIGH|MEDIUM|LOW)\b/gi;

  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(text.substring(lastIndex, match.index));
    }

    const matchedStr = match[0];
    const upper = matchedStr.toUpperCase();
    let badgeClass = "bg-neutral-100 text-black border-neutral-300";

    if (upper.includes("CRITICAL")) {
      badgeClass = "bg-red-50 text-red-700 border-red-300 font-bold";
    } else if (upper.includes("HIGH")) {
      badgeClass = "bg-orange-50 text-orange-700 border-orange-300 font-bold";
    } else if (upper.includes("MEDIUM")) {
      badgeClass = "bg-amber-50 text-amber-700 border-amber-300 font-medium";
    } else if (upper.includes("LOW")) {
      badgeClass = "bg-emerald-50 text-emerald-700 border-emerald-300 font-medium";
    }

    parts.push(
      <span
        key={match.index}
        className={`inline-flex items-center px-2 py-0.5 text-[11px] font-mono border rounded uppercase tracking-wider mx-1 ${badgeClass}`}
      >
        {matchedStr}
      </span>
    );
    lastIndex = regex.lastIndex;
  }

  if (lastIndex < text.length) {
    parts.push(text.substring(lastIndex));
  }

  return parts;
}

function sanitizeLegalMarkdown(rawText) {
  if (!rawText) return "";
  let text = rawText;

  text = text.replace(/\r\n/g, "\n").replace(/\n{3,}/g, "\n\n");

  text = text.replace(/^(\s*[\*\-\d\.]*\s*)\*\*([^*\n:]+):?(\s*)$/gm, "$1**$2:**$3");

  text = text.replace(/\*{3,}([^*]+)\*{3,}/g, "**$1**");

  text = text.replace(/^(\s*)\*([^\s\*])/gm, "$1* $2");

  text = text.replace(/\*\*\s*\*\*/g, "");

  text = text.replace(/^(#{1,6})([^\s#])/gm, "$1 $2");

  return text.trim();
}

function cleanVisibleAsterisks(val) {
  if (typeof val !== "string") return val;
  const noAsterisks = val.replace(/\*{1,3}/g, "");
  return renderRiskBadge(noAsterisks);
}

export function LegalMarkdown({ content = "", className = "" }) {
  if (!content) return null;

  try {
    const sanitized = sanitizeLegalMarkdown(content);

    return (
      <div className={`legal-markdown text-neutral-900 leading-relaxed font-sans ${className}`}>
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
        components={{

          h1: ({ node, children, ...props }) => (
            <h1
              className="font-serif text-xl sm:text-2xl font-bold text-black mt-4 mb-2 pb-1 border-b border-neutral-200 tracking-tight"
              {...props}
            >
              {children}
            </h1>
          ),
          h2: ({ node, children, ...props }) => (
            <h2
              className="font-serif text-lg sm:text-xl font-bold text-black mt-3 mb-2 tracking-tight"
              {...props}
            >
              {children}
            </h2>
          ),
          h3: ({ node, children, ...props }) => (
            <h3
              className="font-serif text-base sm:text-lg font-semibold text-black mt-3 mb-1.5 tracking-tight"
              {...props}
            >
              {children}
            </h3>
          ),
          h4: ({ node, children, ...props }) => (
            <h4
              className="font-mono text-sm font-bold text-black mt-2 mb-1 uppercase tracking-wider"
              {...props}
            >
              {children}
            </h4>
          ),

          p: ({ node, children, ...props }) => (
            <p className="mb-3 last:mb-0 text-sm sm:text-base leading-relaxed text-neutral-800" {...props}>
              {React.Children.map(children, (child) =>
                typeof child === "string" ? cleanVisibleAsterisks(child) : child
              )}
            </p>
          ),

          strong: ({ node, children, ...props }) => (
            <strong className="font-bold text-black" {...props}>
              {children}
            </strong>
          ),

          em: ({ node, children, ...props }) => (
            <em className="italic text-neutral-900" {...props}>
              {children}
            </em>
          ),

          ul: ({ node, children, ...props }) => (
            <ul className="list-disc pl-5 my-2.5 space-y-1.5 text-sm sm:text-base text-neutral-800" {...props}>
              {children}
            </ul>
          ),

          ol: ({ node, children, ...props }) => (
            <ol className="list-decimal pl-5 my-2.5 space-y-1.5 text-sm sm:text-base text-neutral-800" {...props}>
              {children}
            </ol>
          ),

          li: ({ node, children, ...props }) => (
            <li className="leading-relaxed pl-1" {...props}>
              {React.Children.map(children, (child) =>
                typeof child === "string" ? cleanVisibleAsterisks(child) : child
              )}
            </li>
          ),

          blockquote: ({ node, children, ...props }) => (
            <blockquote
              className="border-l-4 border-black bg-neutral-50 px-4 py-2 my-3 italic text-neutral-700 rounded-r text-sm sm:text-base"
              {...props}
            >
              {children}
            </blockquote>
          ),

          table: ({ node, children, ...props }) => (
            <div className="overflow-x-auto my-4 border border-neutral-300 rounded shadow-sm">
              <table className="min-w-full divide-y divide-neutral-200 text-xs sm:text-sm text-left font-sans" {...props}>
                {children}
              </table>
            </div>
          ),
          thead: ({ node, children, ...props }) => (
            <thead className="bg-neutral-100 border-b border-neutral-300" {...props}>
              {children}
            </thead>
          ),
          tbody: ({ node, children, ...props }) => (
            <tbody className="divide-y divide-neutral-200 bg-white" {...props}>
              {children}
            </tbody>
          ),
          tr: ({ node, children, ...props }) => (
            <tr className="hover:bg-neutral-50 transition-colors" {...props}>
              {children}
            </tr>
          ),
          th: ({ node, children, ...props }) => (
            <th className="px-3.5 py-2.5 font-bold text-black uppercase tracking-wider text-xs border-r border-neutral-200 last:border-r-0" {...props}>
              {children}
            </th>
          ),
          td: ({ node, children, ...props }) => (
            <td className="px-3.5 py-2 text-neutral-800 border-r border-neutral-200 last:border-r-0 leading-normal" {...props}>
              {children}
            </td>
          ),

          code: ({ node, inline, className, children, ...props }) => {
            if (inline) {
              return (
                <code
                  className="font-mono bg-neutral-100 text-neutral-900 px-1.5 py-0.5 rounded text-xs border border-neutral-200"
                  {...props}
                >
                  {children}
                </code>
              );
            }
            return (
              <pre className="font-mono bg-neutral-900 text-neutral-100 p-3.5 rounded-lg overflow-x-auto text-xs my-3 border border-neutral-800">
                <code {...props}>{children}</code>
              </pre>
            );
          },

          a: ({ node, href, children, ...props }) => (
            <a
              href={sanitizeUrl(href)}
              target="_blank"
              rel="noopener noreferrer"
              className="text-black font-semibold underline underline-offset-2 hover:text-neutral-600 transition-colors"
              {...props}
            >
              {children}
            </a>
          ),

          hr: ({ node, ...props }) => (
            <hr className="my-4 border-t border-neutral-200" {...props} />
          ),
        }}
      >
        {sanitized}
      </ReactMarkdown>
    </div>
  );
  } catch (err) {
    console.error("LegalMarkdown render fallback:", err);
    return (
      <div className={`legal-markdown text-neutral-900 leading-relaxed font-sans whitespace-pre-wrap ${className}`}>
        {content}
      </div>
    );
  }
}

export default LegalMarkdown;
