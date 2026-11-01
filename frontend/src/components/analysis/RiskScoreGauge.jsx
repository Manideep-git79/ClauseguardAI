import React from "react";
import { ResponsiveContainer, PieChart, Pie, Cell } from "recharts";

export function RiskScoreGauge({ score = 0, band = "low" }) {
  const normalizedScore = Math.min(100, Math.max(0, Number(score) || 0));

  const bandColors = {
    low: "#22C55E",
    medium: "#F59E0B",
    high: "#F97316",
    critical: "#EF4444",
    insufficient_content: "#6B7280",
  };

  const currentColor = bandColors[band.toLowerCase()] || bandColors.medium;
  const isInsufficient = band.toLowerCase() === "insufficient_content";

  const data = [
    { name: "Score", value: normalizedScore },
    { name: "Remaining", value: 100 - normalizedScore },
  ];

  return (
    <div className="border border-neutral-200 bg-white p-6 flex flex-col justify-between shadow-sm">
      <div className="flex items-center justify-between border-b border-neutral-200 pb-3 mb-2">
        <span className="text-[11px] font-mono uppercase tracking-widest text-neutral-600 font-semibold">
          Composite Risk Assessment
        </span>
        <span
          className="text-xs font-mono font-bold uppercase px-2 py-0.5 border"
          style={{ borderColor: currentColor, color: currentColor }}
        >
          {isInsufficient ? "Insufficient Content" : `${band} Risk`}
        </span>
      </div>

      <div className="relative h-44 w-full flex items-center justify-center">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="75%"
              startAngle={180}
              endAngle={0}
              innerRadius={65}
              outerRadius={90}
              paddingAngle={2}
              dataKey="value"
              stroke="none"
            >
              <Cell fill={currentColor} />
              <Cell fill="rgba(0, 0, 0, 0.08)" />
            </Pie>
          </PieChart>
        </ResponsiveContainer>

        <div className="absolute top-[52%] left-1/2 -translate-x-1/2 -translate-y-1/2 text-center pointer-events-none">
          <span className="font-serif text-4xl font-bold tracking-tight text-black block">
            {normalizedScore}
          </span>
          <span className="text-[10px] font-mono text-neutral-500 uppercase tracking-widest block -mt-1 font-semibold">
            out of 100
          </span>
        </div>
      </div>

      <div className="grid grid-cols-4 gap-1 text-[10px] font-mono text-center pt-3 border-t border-neutral-200">
        <div className="p-1 border border-green-300 bg-green-50 text-green-700">
          <span className="font-bold">0-25</span>
          <span className="block text-[9px] text-neutral-500">LOW</span>
        </div>
        <div className="p-1 border border-amber-300 bg-amber-50 text-amber-700">
          <span className="font-bold">26-50</span>
          <span className="block text-[9px] text-neutral-500">MED</span>
        </div>
        <div className="p-1 border border-orange-300 bg-orange-50 text-orange-700">
          <span className="font-bold">51-75</span>
          <span className="block text-[9px] text-neutral-500">HIGH</span>
        </div>
        <div className="p-1 border border-red-300 bg-red-50 text-red-700">
          <span className="font-bold">76-100</span>
          <span className="block text-[9px] text-neutral-500">CRIT</span>
        </div>
      </div>
    </div>
  );
}
