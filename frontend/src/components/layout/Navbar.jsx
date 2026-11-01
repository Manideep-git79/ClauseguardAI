import { ShieldAlert, FileText, UploadCloud, Bot, LogOut, LogIn, UserPlus, User, GitFork } from "lucide-react";
import { useAuth } from "../../context/AuthContext";

export function Navbar({ currentView, onNavigate }) {
  const { user, isAuthenticated, logout } = useAuth();

  const handleLogout = async () => {
    await logout();
    if (onNavigate) {
      onNavigate("landing");
    }
  };

  return (
    <header className="border-b border-neutral-200 bg-white sticky top-0 z-40">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        <button
          onClick={() => onNavigate("landing")}
          className="flex items-center gap-3 text-left group"
        >
          <div className="w-8 h-8 border border-neutral-300 bg-neutral-100 flex items-center justify-center text-black group-hover:border-black transition-colors rounded">
            <ShieldAlert className="w-4 h-4 text-black" />
          </div>
          <div>
            <span className="font-serif text-lg font-bold tracking-tight text-black block">
              ClauseGuard AI
            </span>
            <span className="text-[10px] font-mono text-neutral-500 tracking-wider uppercase block -mt-1">
              Legal Risk Analyzer & Copilot
            </span>
          </div>
        </button>

        <nav className="flex items-center gap-1 sm:gap-2">
          <button
            onClick={() => onNavigate("landing")}
            className={`px-3 py-2 text-xs font-mono uppercase tracking-wider transition-colors border rounded ${
              currentView === "landing"
                ? "border-black text-black bg-neutral-100 font-semibold"
                : "border-transparent text-neutral-600 hover:text-black hover:border-neutral-200"
            }`}
          >
            Overview
          </button>

          <button
            onClick={() => onNavigate("documents")}
            className={`px-3 py-2 text-xs font-mono uppercase tracking-wider transition-colors border rounded ${
              currentView === "documents"
                ? "border-black text-black bg-neutral-100 font-semibold"
                : "border-transparent text-neutral-600 hover:text-black hover:border-neutral-200"
            }`}
          >
            Documents
          </button>

          <button
            onClick={() => onNavigate("chat")}
            className={`px-3 py-2 text-xs font-mono uppercase tracking-wider transition-colors border rounded flex items-center gap-1.5 ${
              currentView === "chat"
                ? "border-black text-black bg-neutral-100 font-semibold"
                : "border-transparent text-neutral-600 hover:text-black hover:border-neutral-200"
            }`}
          >
            <Bot className="w-3.5 h-3.5 text-black" />
            <span>Deciva AI</span>
          </button>

          <button
            onClick={() => onNavigate("architecture")}
            className={`px-3 py-2 text-xs font-mono uppercase tracking-wider transition-colors border rounded flex items-center gap-1.5 ${
              currentView === "architecture"
                ? "border-black text-black bg-neutral-100 font-semibold"
                : "border-transparent text-neutral-600 hover:text-black hover:border-neutral-200"
            }`}
          >
            <GitFork className="w-3.5 h-3.5 text-black" />
            <span className="hidden md:inline">Architecture</span>
            <span className="md:hidden">Flow</span>
          </button>

          <button
            onClick={() => onNavigate("upload")}
            className="px-3.5 py-2 text-xs font-mono uppercase tracking-wider border border-black text-white bg-black hover:bg-neutral-800 transition-colors flex items-center gap-1.5 font-semibold rounded"
          >
            <UploadCloud className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Upload Document</span>
            <span className="sm:hidden">Upload</span>
          </button>

          <div className="h-6 w-px bg-neutral-200 mx-1 hidden sm:block" />

          {isAuthenticated && user ? (
            <div className="flex items-center gap-2">
              <div className="hidden md:flex items-center gap-2 px-2.5 py-1.5 border border-neutral-200 bg-neutral-50 text-xs font-mono rounded">
                <User className="w-3.5 h-3.5 text-black" />
                <span className="text-black max-w-[120px] truncate">{user.name || user.email}</span>
                <span className="text-[10px] uppercase px-1 py-0.2 bg-neutral-100 border border-neutral-300 text-neutral-700 rounded font-semibold">
                  {user.role || "user"}
                </span>
              </div>
              <button
                onClick={handleLogout}
                title="Sign Out"
                className="p-2 text-xs font-mono border border-transparent hover:border-neutral-300 text-neutral-600 hover:text-black hover:bg-neutral-100 transition-colors flex items-center gap-1 rounded"
              >
                <LogOut className="w-3.5 h-3.5" />
                <span className="hidden lg:inline text-[11px] uppercase tracking-wider">Sign Out</span>
              </button>
            </div>
          ) : (
            <div className="flex items-center gap-1.5">
              <button
                onClick={() => onNavigate("login")}
                className={`px-3 py-1.5 text-xs font-mono uppercase tracking-wider border rounded transition-colors flex items-center gap-1.5 ${
                  currentView === "login"
                    ? "border-black text-white bg-black font-semibold"
                    : "border-neutral-200 text-neutral-700 hover:text-black hover:border-black"
                }`}
              >
                <LogIn className="w-3.5 h-3.5" />
                <span>Sign In</span>
              </button>
              <button
                onClick={() => onNavigate("register")}
                className="hidden sm:flex px-3 py-1.5 text-xs font-mono uppercase tracking-wider border border-black bg-black text-white hover:bg-neutral-800 transition-colors items-center gap-1.5 font-semibold rounded"
              >
                <UserPlus className="w-3.5 h-3.5" />
                <span>Register</span>
              </button>
            </div>
          )}
        </nav>
      </div>
    </header>
  );
}

export default Navbar;
