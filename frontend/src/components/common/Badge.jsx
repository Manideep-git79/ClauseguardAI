import React from "react";

export function Badge({ variant = "default", children, className = "" }) {
  const variantStyles = {

    low: "border-green-300 bg-green-50 text-green-800 font-semibold",
    medium: "border-amber-300 bg-amber-50 text-amber-800 font-semibold",
    high: "border-orange-300 bg-orange-50 text-orange-800 font-semibold",
    critical: "border-red-300 bg-red-50 text-red-800 font-semibold",

    fair: "border-green-300 bg-green-50 text-green-800 font-semibold",
    needs_review: "border-amber-300 bg-amber-50 text-amber-800 font-semibold",
    unfavorable: "border-red-300 bg-red-50 text-red-800 font-semibold",

    info: "border-neutral-300 bg-neutral-100 text-black font-semibold",
    insufficient_content: "border-neutral-300 bg-neutral-100 text-neutral-700 font-semibold",
    default: "border-neutral-300 bg-neutral-100 text-neutral-800 font-semibold",
  };

  const style = variantStyles[variant.toLowerCase()] || variantStyles.default;

  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 text-[11px] font-mono uppercase tracking-wider border ${style} ${className}`}
    >
      {children}
    </span>
  );
}
