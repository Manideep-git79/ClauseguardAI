import React, { useState } from "react";
import {
  ShieldAlert,
  Shield,
  Layers,
  FileText,
  UploadCloud,
  Bot,
  Activity,
  BookOpen,
  GitFork,
  LogOut,
  User,
  ChevronRight,
  Menu,
  X,
  Sparkles,
} from "lucide-react";
import { useAuth } from "../../context/AuthContext";

export function Sidebar({ currentView, onNavigate }) {
  const { user, logout } = useAuth();
  const [mobileOpen, setMobileOpen] = useState(false);

  const navItems = [
    {
      group: "WORKSPACE",
      items: [
        { id: "landing", label: "Overview", icon: Layers, path: "/" },
        { id: "documents", label: "Documents", icon: FileText, path: "/documents" },
        { id: "upload", label: "Upload Document", icon: UploadCloud, path: "/upload" },
        {
          id: "chat",
          label: "Deciva AI",
          icon: Bot,
          path: "/chat",
          badge: "Deciva",
          badgeColor: "bg-neutral-100 text-neutral-800 border-neutral-300",
        },
      ],
    },
    {
      group: "GOVERNANCE & HELP",
      items: [
        {
          id: "architecture",
          label: "Architecture Flow",
          icon: GitFork,
          path: "/architecture/flow",
          badge: "Code",
          badgeColor: "bg-neutral-100 text-neutral-800 border-neutral-300",
        },
        {
          id: "audit",
          label: "Audit & Trail",
          icon: Activity,
          path: "/audit",
          badge: "Logs",
          badgeColor: "bg-neutral-100 text-neutral-800 border-neutral-300",
        },
        {
          id: "guide",
          label: "Platform Guide",
          icon: BookOpen,
          path: "/guide",
          badge: "Guide",
          badgeColor: "bg-neutral-100 text-neutral-800 border-neutral-300",
        },
      ],
    },
  ];

  const handleNavClick = (viewId) => {
    if (onNavigate) {
      onNavigate(viewId);
    }
    setMobileOpen(false);
  };

  const handleLogout = async () => {
    await logout();
    if (onNavigate) {
      onNavigate("login");
    }
  };

  return (
    <>
      <div className="lg:hidden flex items-center justify-between px-4 py-3 bg-white border-b border-neutral-200 sticky top-0 z-50">
        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 border border-neutral-300 bg-neutral-100 flex items-center justify-center text-black rounded">
            <Shield className="w-4 h-4 text-black" />
          </div>
          <span className="font-serif text-sm font-bold tracking-tight text-black">
            ClauseGuard AI
          </span>
        </div>
        <button
          onClick={() => setMobileOpen(!mobileOpen)}
          className="p-1.5 border border-neutral-300 text-neutral-700 hover:text-black bg-neutral-100 rounded"
        >
          {mobileOpen ? <X size={20} /> : <Menu size={20} />}
        </button>
      </div>

      {mobileOpen && (
        <div
          onClick={() => setMobileOpen(false)}
          className="lg:hidden fixed inset-0 bg-black/40 backdrop-blur-sm z-40"
        />
      )}

      <aside
        className={`fixed top-0 bottom-0 left-0 z-50 w-64 sm:w-72 bg-white border-r border-neutral-200 flex flex-col transition-transform duration-200 ease-in-out ${
          mobileOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0"
        }`}
      >
        <div className="p-5 border-b border-neutral-200 bg-white">
          <button
            onClick={() => handleNavClick("landing")}
            className="flex items-center gap-3 text-left w-full group"
          >
            <div className="w-9 h-9 rounded-lg border border-neutral-300 bg-neutral-100 flex items-center justify-center text-black group-hover:border-black transition-colors">
              <Shield className="w-5 h-5 text-black" />
            </div>
            <div>
              <span className="font-serif text-base font-bold tracking-tight text-black block">
                ClauseGuard AI
              </span>
              <span className="text-[10px] font-mono text-neutral-500 tracking-wider uppercase block -mt-0.5">
                Enterprise Legal AI
              </span>
            </div>
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-3 py-5 space-y-6">
          {navItems.map((group, gIdx) => (
            <div key={gIdx} className="space-y-1.5">
              <div className="px-3 text-[10px] font-mono text-neutral-500 tracking-widest uppercase">
                {group.group}
              </div>
              <div className="space-y-1">
                {group.items.map((item) => {
                  const Icon = item.icon;
                  const isActive = currentView === item.id;
                  return (
                    <button
                      key={item.id}
                      onClick={() => handleNavClick(item.id)}
                      className={`w-full flex items-center justify-between px-3 py-2.5 rounded-md text-xs font-mono transition-all ${
                        isActive
                          ? "bg-black text-white font-semibold shadow-sm"
                          : "text-neutral-700 hover:text-black hover:bg-neutral-100 border border-transparent"
                      }`}
                    >
                      <div className="flex items-center gap-2.5">
                        <Icon
                          className={`w-4 h-4 ${
                            isActive ? "text-white" : "text-neutral-500"
                          }`}
                        />
                        <span>{item.label}</span>
                      </div>
                      {item.badge && (
                        <span
                          className={`text-[9px] font-mono border px-1.5 py-0.5 rounded tracking-wide ${
                            isActive ? "bg-neutral-800 text-white border-neutral-700" : item.badgeColor
                          }`}
                        >
                          {item.badge}
                        </span>
                      )}
                    </button>
                  );
                })}
              </div>
            </div>
          ))}
        </div>

        <div className="p-4 border-t border-neutral-200 bg-white">
          {user ? (
            <div className="space-y-3">
              <div className="flex items-center gap-3 px-2 py-1.5 rounded bg-neutral-50 border border-neutral-200">
                <div className="w-8 h-8 rounded-full bg-neutral-100 border border-neutral-300 flex items-center justify-center text-black">
                  <User size={16} />
                </div>
                <div className="flex-1 min-w-0">
                  <span className="text-xs font-mono font-medium text-black block truncate">
                    {user.name || user.email}
                  </span>
                  <span className="text-[10px] font-mono text-neutral-500 block truncate">
                    {user.email}
                  </span>
                </div>
                <span className="text-[9px] font-mono uppercase px-1.5 py-0.5 bg-neutral-100 border border-neutral-300 text-neutral-700 rounded font-semibold">
                  {user.role || "user"}
                </span>
              </div>

              <button
                onClick={handleLogout}
                className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded text-xs font-mono text-neutral-700 hover:text-black hover:bg-neutral-100 border border-neutral-200 transition-colors"
              >
                <LogOut size={14} />
                <span>Sign Out</span>
              </button>
            </div>
          ) : (
            <button
              onClick={() => handleNavClick("login")}
              className="w-full py-2 bg-black hover:bg-neutral-800 text-white font-semibold text-xs font-mono rounded transition-colors"
            >
              Sign In to Account
            </button>
          )}
        </div>
      </aside>
    </>
  );
}

export default Sidebar;
