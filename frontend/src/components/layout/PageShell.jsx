import React from "react";
import { Sidebar } from "./Sidebar";

export function PageShell({ children, currentView, onNavigate }) {
  return (
    <div className="min-h-screen bg-white text-black flex flex-col selection:bg-black selection:text-white">
      <Sidebar currentView={currentView} onNavigate={onNavigate} />

      <div className="lg:pl-72 flex-1 flex flex-col min-w-0">
        <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {children}
        </main>

        <footer className="border-t border-neutral-200 bg-white py-8 mt-auto">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-4">
            <div className="flex flex-col sm:flex-row items-center justify-between text-xs font-mono text-neutral-600 gap-4">
              <div>
                <span>ClauseGuard AI © 2026. Enterprise Legal Risk Intelligence & Deciva AI.</span>
              </div>
              <div className="flex gap-4 text-neutral-600">
                <span>Residential Leases</span>
                <span>•</span>
                <span>Employment Contracts</span>
                <span>•</span>
                <span>Insurance Policies</span>
              </div>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
}

export default PageShell;
