import React, { useState } from "react";
import { useAuth } from "../context/AuthContext";
import { api } from "../api/client";
import {
  ShieldAlert,
  ShieldCheck,
  Lock,
  Mail,
  User,
  Eye,
  EyeOff,
  ArrowRight,
  AlertCircle,
} from "lucide-react";

export function AuthPortal({ onAuthSuccess }) {
  const [tab, setTab] = useState("login");
  const { login } = useAuth();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");

  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);

  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const handleLoginSubmit = async (e) => {
    e.preventDefault();
    setError("");

    if (!email.trim() || !password) {
      setError("Please enter your email and password.");
      return;
    }

    setSubmitting(true);
    try {
      const data = await api.login(email.trim().toLowerCase(), password);

      if (data.token) {
        try {
          localStorage.setItem("cg_token", data.token);
        } catch (e) {}
      }

      login(data.user);
      if (onAuthSuccess) onAuthSuccess(data.user);
    } catch (err) {
      setError(err.message || "Failed to sign in. Please verify your credentials.");
    } finally {
      setSubmitting(false);
    }
  };

  const handleRegisterSubmit = async (e) => {
    e.preventDefault();
    setError("");

    if (!name.trim()) return setError("Please enter your full name");
    if (!email.trim()) return setError("Please enter your email address");
    if (password.length < 8) return setError("Password must be at least 8 characters");
    if (password !== confirmPassword) return setError("Passwords do not match");

    setSubmitting(true);
    try {
      const data = await api.register({
        name: name.trim(),
        email: email.trim().toLowerCase(),
        password,
        confirmPassword,
      });

      if (data.token) {
        try {
          localStorage.setItem("cg_token", data.token);
        } catch (e) {}
      }

      login(data.user);
      if (onAuthSuccess) onAuthSuccess(data.user);
    } catch (err) {
      setError(err.message || "Failed to create account. Please check your details.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-white text-black flex flex-col justify-center items-center px-4 py-12 relative overflow-hidden">
      <div className="w-full max-w-md relative z-10 space-y-6">
        <div className="text-center space-y-2">
          <div className="inline-flex items-center justify-center w-12 h-12 rounded-xl bg-neutral-100 border border-neutral-300 text-black mb-2 shadow-sm">
            <ShieldAlert size={26} />
          </div>
          <h1 className="font-serif text-3xl font-bold tracking-tight text-black">
            ClauseGuard AI
          </h1>
          <p className="text-xs font-mono text-neutral-500 tracking-wide uppercase">
            Enterprise Legal Risk Analyzer & Deciva AI
          </p>
        </div>

        <div className="bg-white border border-neutral-200 rounded-2xl p-6 sm:p-8 shadow-xl space-y-6">
          <div className="flex bg-neutral-100 p-1 rounded-lg border border-neutral-200">
            <button
              onClick={() => {
                setTab("login");
                setError("");
              }}
              className={`flex-1 py-2 text-xs font-mono font-medium rounded-md transition-all ${
                tab === "login"
                  ? "bg-black text-white font-semibold shadow-sm"
                  : "text-neutral-600 hover:text-black"
              }`}
            >
              Sign In
            </button>
            <button
              onClick={() => {
                setTab("register");
                setError("");
              }}
              className={`flex-1 py-2 text-xs font-mono font-medium rounded-md transition-all ${
                tab === "register"
                  ? "bg-black text-white font-semibold shadow-sm"
                  : "text-neutral-600 hover:text-black"
              }`}
            >
              Create Account
            </button>
          </div>

          {error && (
            <div className="p-3 bg-red-50 border border-red-200 rounded-lg text-xs font-mono text-red-700 flex items-center gap-2">
              <AlertCircle size={15} className="shrink-0 text-red-600" />
              <span>{error}</span>
            </div>
          )}

          {tab === "login" ? (
            <form onSubmit={handleLoginSubmit} className="space-y-4">
              <div>
                <label className="block text-xs font-mono text-neutral-700 font-medium mb-1.5">
                  Email Address / Username
                </label>
                <div className="relative">
                  <Mail size={16} className="absolute left-3.5 top-3 text-neutral-400" />
                  <input
                    type="email"
                    required
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="you@company.com"
                    className="w-full bg-neutral-50 border border-neutral-300 rounded-lg pl-10 pr-4 py-2.5 text-xs font-mono text-black placeholder-neutral-400 focus:outline-none focus:border-black transition-colors"
                  />
                </div>
              </div>

              <div>
                <label className="block text-xs font-mono text-neutral-700 font-medium mb-1.5">
                  Password
                </label>
                <div className="relative">
                  <Lock size={16} className="absolute left-3.5 top-3 text-neutral-400" />
                  <input
                    type={showPassword ? "text" : "password"}
                    required
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="••••••••••••"
                    className="w-full bg-neutral-50 border border-neutral-300 rounded-lg pl-10 pr-10 py-2.5 text-xs font-mono text-black placeholder-neutral-400 focus:outline-none focus:border-black transition-colors"
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-3 top-2.5 text-neutral-400 hover:text-black transition-colors"
                    title={showPassword ? "Hide password" : "Show password"}
                  >
                    {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
                  </button>
                </div>
              </div>

              <button
                type="submit"
                disabled={submitting}
                className="w-full py-2.5 bg-black hover:bg-neutral-800 text-white text-xs font-mono font-bold rounded-lg transition-colors flex items-center justify-center gap-2 border border-black disabled:bg-neutral-200 disabled:text-neutral-400 disabled:cursor-not-allowed mt-2"
              >
                {submitting ? (
                  <span>Authenticating…</span>
                ) : (
                  <>
                    <span>Sign In to Workspace</span>
                    <ArrowRight size={14} />
                  </>
                )}
              </button>
            </form>
          ) : (

            <form onSubmit={handleRegisterSubmit} className="space-y-4">
              <div>
                <label className="block text-xs font-mono text-neutral-700 font-medium mb-1.5">
                  Full Name
                </label>
                <div className="relative">
                  <User size={16} className="absolute left-3.5 top-3 text-neutral-400" />
                  <input
                    type="text"
                    required
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    placeholder="Alex Morgan"
                    className="w-full bg-neutral-50 border border-neutral-300 rounded-lg pl-10 pr-4 py-2.5 text-xs font-mono text-black placeholder-neutral-400 focus:outline-none focus:border-black transition-colors"
                  />
                </div>
              </div>

              <div>
                <label className="block text-xs font-mono text-neutral-700 font-medium mb-1.5">
                  Email Address
                </label>
                <div className="relative">
                  <Mail size={16} className="absolute left-3.5 top-3 text-neutral-400" />
                  <input
                    type="email"
                    required
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="alex@company.com"
                    className="w-full bg-neutral-50 border border-neutral-300 rounded-lg pl-10 pr-4 py-2.5 text-xs font-mono text-black placeholder-neutral-400 focus:outline-none focus:border-black transition-colors"
                  />
                </div>
              </div>

              <div>
                <label className="block text-xs font-mono text-neutral-700 font-medium mb-1.5">
                  Password (min 8 chars)
                </label>
                <div className="relative">
                  <Lock size={16} className="absolute left-3.5 top-3 text-neutral-400" />
                  <input
                    type={showPassword ? "text" : "password"}
                    required
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="••••••••••••"
                    className="w-full bg-neutral-50 border border-neutral-300 rounded-lg pl-10 pr-10 py-2.5 text-xs font-mono text-black placeholder-neutral-400 focus:outline-none focus:border-black transition-colors"
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-3 top-2.5 text-neutral-400 hover:text-black transition-colors"
                    title={showPassword ? "Hide password" : "Show password"}
                  >
                    {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
                  </button>
                </div>
              </div>

              <div>
                <label className="block text-xs font-mono text-neutral-700 font-medium mb-1.5">
                  Confirm Password
                </label>
                <div className="relative">
                  <Lock size={16} className="absolute left-3.5 top-3 text-neutral-400" />
                  <input
                    type={showConfirmPassword ? "text" : "password"}
                    required
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    placeholder="••••••••••••"
                    className="w-full bg-neutral-50 border border-neutral-300 rounded-lg pl-10 pr-10 py-2.5 text-xs font-mono text-black placeholder-neutral-400 focus:outline-none focus:border-black transition-colors"
                  />
                  <button
                    type="button"
                    onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                    className="absolute right-3 top-2.5 text-neutral-400 hover:text-black transition-colors"
                    title={showConfirmPassword ? "Hide password" : "Show password"}
                  >
                    {showConfirmPassword ? <EyeOff size={16} /> : <Eye size={16} />}
                  </button>
                </div>
              </div>

              <button
                type="submit"
                disabled={submitting}
                className="w-full py-2.5 bg-black hover:bg-neutral-800 text-white text-xs font-mono font-bold rounded-lg transition-colors flex items-center justify-center gap-2 border border-black disabled:bg-neutral-200 disabled:text-neutral-400 disabled:cursor-not-allowed mt-2"
              >
                {submitting ? (
                  <span>Creating Account…</span>
                ) : (
                  <>
                    <span>Register & Access Platform</span>
                    <ArrowRight size={14} />
                  </>
                )}
              </button>
            </form>
          )}

          <div className="pt-4 border-t border-neutral-200 flex items-center justify-between text-[10px] font-mono text-neutral-500">
            <span className="flex items-center gap-1 text-neutral-600 font-medium">
              <ShieldCheck size={12} />
              Bcrypt-12 Salting
            </span>
            <span>•</span>
            <span>HTTP-Only JWT</span>
            <span>•</span>
            <span>Zero-PII Redaction</span>
          </div>
        </div>
      </div>
    </div>
  );
}

export default AuthPortal;
