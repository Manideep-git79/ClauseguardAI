import React, { useRef, useState } from "react";
import { Upload, FileText, CheckCircle2, AlertCircle, X } from "lucide-react";

export function Dropzone({ file, onFileSelect, onClearFile, error }) {
  const [isDragOver, setIsDragOver] = useState(false);
  const inputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      onFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      onFileSelect(e.target.files[0]);
    }
  };

  return (
    <div className="space-y-3">
      <label className="block text-xs font-mono uppercase tracking-wider text-neutral-600 font-semibold">
        2. Upload Document File <span className="text-neutral-500">(.pdf, .docx, .txt, .png, .jpg — Max 20MB)</span>
      </label>

      {!file ? (
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => inputRef.current?.click()}
          className={`border-2 border-dashed p-10 text-center cursor-pointer transition-all ${
            isDragOver
              ? "border-black bg-neutral-100"
              : error
              ? "border-red-500 bg-red-50"
              : "border-neutral-300 hover:border-black bg-neutral-50"
          }`}
        >
          <input
            ref={inputRef}
            type="file"
            accept=".pdf,.docx,.txt,.png,.jpg,.jpeg"
            onChange={handleFileChange}
            className="hidden"
          />
          <div className="w-12 h-12 border border-neutral-300 bg-white mx-auto mb-4 flex items-center justify-center text-black shadow-sm">
            <Upload className="w-5 h-5" />
          </div>
          <p className="font-serif text-lg text-black font-bold mb-1">
            Drag & drop legal document here
          </p>
          <p className="text-xs text-neutral-600 font-sans mb-4">
            or click to browse your computer
          </p>
          <span className="inline-block text-[11px] font-mono text-neutral-600 border border-neutral-300 bg-white px-3 py-1">
            DIGITAL OR SCANNED PDF • WORD DOCX • PLAIN TXT • IMAGES (PNG, JPG)
          </span>

          {error && (
            <p className="text-xs text-red-600 mt-3 flex items-center justify-center gap-1.5 font-mono">
              <AlertCircle className="w-3.5 h-3.5" />
              {error}
            </p>
          )}
        </div>
      ) : (
        <div className="border border-neutral-300 bg-white p-4 flex items-center justify-between shadow-sm">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 border border-neutral-300 bg-neutral-100 flex items-center justify-center text-black">
              <FileText className="w-5 h-5" />
            </div>
            <div>
              <p className="text-sm font-bold text-black font-mono">{file.name}</p>
              <p className="text-xs text-neutral-600 font-mono">
                {(file.size / (1024 * 1024)).toFixed(2)} MB • {file.name.split(".").pop().toUpperCase()}
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClearFile}
            className="p-1.5 border border-neutral-300 hover:border-black text-neutral-600 hover:text-black transition-colors"
            aria-label="Remove file"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      )}
    </div>
  );
}
