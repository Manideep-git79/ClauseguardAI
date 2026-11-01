import React, { useState } from "react";
import { DocumentTypeSelect } from "../components/upload/DocumentTypeSelect";
import { Dropzone } from "../components/upload/Dropzone";
import { ProcessingStatus } from "../components/upload/ProcessingStatus";
import { usePolling } from "../hooks/usePolling";
import { api } from "../api/client";
import { ArrowLeft, ShieldAlert, ArrowRight, AlertCircle } from "lucide-react";

export function Upload({ onNavigate, initialDocType = "rental_agreement" }) {
  const [documentType, setDocumentType] = useState(initialDocType);
  const [file, setFile] = useState(null);
  const [error, setError] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [processingDocId, setProcessingDocId] = useState(null);

  const { status, processingStage, errorMessage } = usePolling(processingDocId, {
    interval: 1000,
    onComplete: (data) => {

      onNavigate("analysis", { documentId: processingDocId });
    },
    onError: (err) => {
      setError(err || "Processing failed");
    },
    enabled: Boolean(processingDocId),
  });

  const handleFileSelect = (selectedFile) => {
    setError(null);
    const validExtensions = [".pdf", ".docx", ".txt", ".png", ".jpg", ".jpeg"];
    const ext = "." + selectedFile.name.split(".").pop().toLowerCase();

    if (!validExtensions.includes(ext)) {
      setError(`Unsupported file format "${ext}". Supported: .pdf, .docx, .txt, .png, .jpg, .jpeg`);
      return;
    }

    if (selectedFile.size > 20 * 1024 * 1024) {
      setError("File exceeds maximum allowable size of 20 MB.");
      return;
    }

    setFile(selectedFile);
  };

  const handleUpload = async () => {
    if (!file) {
      setError("Please select a document file to upload.");
      return;
    }
    if (!documentType) {
      setError("Please select the document type.");
      return;
    }

    setIsSubmitting(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("document_type", documentType);

    try {
      const response = await api.uploadDocument(formData);
      setProcessingDocId(response.document_id || response.id);
    } catch (err) {
      setError(err.message || "Failed to initiate document upload.");
      setIsSubmitting(false);
    }
  };

  return (
    <div className="space-y-8 max-w-4xl mx-auto py-4">
      <div className="flex items-center justify-between border-b border-neutral-200 pb-4">
        <div>
          <button
            onClick={() => onNavigate("landing")}
            className="inline-flex items-center gap-1.5 text-xs font-mono text-neutral-500 hover:text-black mb-2 transition-colors"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            <span>Back to Overview</span>
          </button>
          <h1 className="font-serif text-3xl font-bold text-black">
            Upload Agreement for AI Audit
          </h1>
          <p className="text-xs text-neutral-600 mt-1 font-sans">
            Specify the document type and provide your PDF, DOCX, or TXT agreement.
          </p>
        </div>

        <button
          onClick={() => onNavigate("documents")}
          className="px-3.5 py-2 text-xs font-mono uppercase tracking-wider border border-neutral-300 hover:border-black text-neutral-700 hover:text-black bg-white transition-colors hidden sm:block rounded font-medium shadow-sm"
        >
          View Library
        </button>
      </div>

      {processingDocId ? (
        <div className="space-y-6">
          <ProcessingStatus
            stage={processingStage}
            status={status}
            error={errorMessage || error}
          />

          {status === "failed" && (
            <div className="text-center pt-4">
              <button
                onClick={() => {
                  setProcessingDocId(null);
                  setIsSubmitting(false);
                }}
                className="px-6 py-2.5 text-xs font-mono uppercase tracking-wider border border-black bg-black text-white hover:bg-neutral-800 rounded font-bold shadow-sm"
              >
                Try Another Document
              </button>
            </div>
          )}
        </div>
      ) : (

        <div className="space-y-8">
          <DocumentTypeSelect value={documentType} onChange={setDocumentType} />

          <Dropzone
            file={file}
            onFileSelect={handleFileSelect}
            onClearFile={() => setFile(null)}
            error={error}
          />

          {error && (
            <div className="border border-red-200 bg-red-50 p-4 text-xs font-mono text-red-700 flex items-start gap-2.5 rounded-lg">
              <AlertCircle className="w-4 h-4 text-red-600 shrink-0 mt-0.5" />
              <span>{error}</span>
            </div>
          )}

          <div className="border border-neutral-200 bg-neutral-50 p-4 text-xs text-neutral-600 font-sans leading-relaxed rounded-lg">
            <span className="font-mono text-neutral-800 uppercase tracking-wider text-[10px] block mb-1 font-semibold">
              Privacy Architecture Note:
            </span>
            Names, phone numbers, email addresses, and residential addresses are detected and masked via Presidio immediately upon text extraction. The raw unredacted text is encrypted at rest and is never sent to any ML classifier or external model.
          </div>

          <div className="flex items-center justify-end gap-3 pt-4 border-t border-neutral-200">
            <button
              type="button"
              onClick={() => onNavigate("landing")}
              className="px-5 py-2.5 text-xs font-mono uppercase tracking-wider border border-neutral-300 hover:border-black text-neutral-600 hover:text-black bg-white transition-colors rounded"
            >
              Cancel
            </button>

            <button
              type="button"
              disabled={!file || !documentType || isSubmitting}
              onClick={handleUpload}
              className="px-8 py-3 text-xs font-mono uppercase tracking-wider border border-black bg-black text-white hover:bg-neutral-800 transition-colors flex items-center gap-2 font-bold disabled:bg-neutral-200 disabled:text-neutral-400 disabled:border-neutral-200 disabled:cursor-not-allowed rounded shadow-sm"
            >
              {isSubmitting ? (
                <>
                  <span className="animate-spin w-3.5 h-3.5 border-2 border-white border-t-transparent inline-block rounded-full" />
                  <span>Uploading & Queuing...</span>
                </>
              ) : (
                <>
                  <span>Begin Risk Audit</span>
                  <ArrowRight className="w-4 h-4" />
                </>
              )}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default Upload;
