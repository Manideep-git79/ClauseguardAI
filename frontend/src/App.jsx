import React from "react";
import {
  BrowserRouter,
  Routes,
  Route,
  Navigate,
  useNavigate,
  useParams,
  useLocation,
} from "react-router-dom";
import { AuthProvider, useAuth } from "./context/AuthContext";
import { PageShell } from "./components/layout/PageShell";
import { Landing } from "./pages/Landing";
import { Upload } from "./pages/Upload";
import { DocumentList } from "./pages/DocumentList";
import { DocumentAnalysis } from "./pages/DocumentAnalysis";
import { StandaloneChat } from "./pages/StandaloneChat";
import { AuditTrail } from "./pages/AuditTrail";
import { PlatformGuide } from "./pages/PlatformGuide";
import { ArchitectureFlow } from "./pages/ArchitectureFlow";
import { AuthPortal } from "./pages/AuthPortal";
import { ErrorBoundary } from "./components/common/ErrorBoundary";
import { ShieldAlert } from "lucide-react";

function DocumentAnalysisWrapper({ onNavigate }) {
  const { id } = useParams();
  return <DocumentAnalysis documentId={id} onNavigate={onNavigate} />;
}

function UploadWrapper({ onNavigate }) {
  const location = useLocation();
  const initialDocType = location.state?.defaultDocType || "rental_agreement";
  return <Upload onNavigate={onNavigate} initialDocType={initialDocType} />;
}

function AuthenticatedApp() {
  const navigate = useNavigate();
  const location = useLocation();

  const getCurrentView = () => {
    const path = location.pathname;
    if (path === "/") return "landing";
    if (path.startsWith("/documents/")) return "analysis";
    if (path === "/documents") return "documents";
    if (path === "/upload") return "upload";
    if (path === "/chat") return "chat";
    if (path === "/audit") return "audit";
    if (path === "/guide") return "guide";
    if (path === "/architecture" || path.startsWith("/architecture/")) return "architecture";
    return "";
  };

  const handleNavigate = (view, params = {}) => {
    switch (view) {
      case "landing":
        navigate("/");
        break;
      case "upload":
        navigate("/upload", { state: params });
        break;
      case "documents":
        navigate("/documents");
        break;
      case "analysis":
        if (params.documentId) {
          navigate(`/documents/${params.documentId}`);
        } else {
          navigate("/documents");
        }
        break;
      case "chat":
        navigate("/chat");
        break;
      case "audit":
        navigate("/audit");
        break;
      case "guide":
        navigate("/guide");
        break;
      case "architecture":
        navigate("/architecture/flow");
        break;
      default:
        navigate("/");
    }
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const currentView = getCurrentView();

  return (
    <PageShell currentView={currentView} onNavigate={handleNavigate}>
      <ErrorBoundary>
        <Routes>
          <Route path="/" element={<Landing onNavigate={handleNavigate} />} />
          <Route path="/upload" element={<UploadWrapper onNavigate={handleNavigate} />} />
          <Route path="/documents" element={<DocumentList onNavigate={handleNavigate} />} />
          <Route path="/documents/:id" element={<DocumentAnalysisWrapper onNavigate={handleNavigate} />} />
          <Route path="/chat" element={<StandaloneChat onNavigate={handleNavigate} />} />
          <Route path="/audit" element={<AuditTrail onNavigate={handleNavigate} />} />
          <Route path="/guide" element={<PlatformGuide onNavigate={handleNavigate} />} />
          <Route path="/architecture" element={<Navigate to="/architecture/flow" replace />} />
          <Route path="/architecture/flow" element={<ArchitectureFlow onNavigate={handleNavigate} />} />
          <Route path="/dashboard" element={<Navigate to="/documents" replace />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </ErrorBoundary>
    </PageShell>
  );
}

function RootController() {
  const { isAuthenticated, loading } = useAuth();
  const navigate = useNavigate();

  if (loading) {
    return (
      <div className="min-h-screen bg-white text-black flex flex-col items-center justify-center space-y-4">
        <div className="w-12 h-12 rounded-xl bg-neutral-100 border border-neutral-300 flex items-center justify-center text-black animate-pulse">
          <ShieldAlert size={26} />
        </div>
        <div className="text-xs font-mono text-neutral-600">
          Verifying security session credentials…
        </div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return <AuthPortal onAuthSuccess={() => navigate("/")} />;
  }

  return <AuthenticatedApp />;
}

export function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <RootController />
      </AuthProvider>
    </BrowserRouter>
  );
}

export default App;
