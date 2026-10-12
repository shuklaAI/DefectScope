import React, { createContext, useContext, useState } from "react";

const InspectionContext = createContext(null);

export function InspectionProvider({ children }) {
  const [pendingFile, setPendingFile] = useState(null);
  const [pendingPreviewUrl, setPendingPreviewUrl] = useState(null);
  const [mode, setMode] = useState("demo"); // "demo" | "real"
  const [currentResult, setCurrentResult] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);

  const reset = () => {
    setPendingFile(null);
    setPendingPreviewUrl(null);
    setCurrentResult(null);
    setError(null);
  };

  const value = {
    pendingFile, setPendingFile,
    pendingPreviewUrl, setPendingPreviewUrl,
    mode, setMode,
    currentResult, setCurrentResult,
    isProcessing, setIsProcessing,
    error, setError,
    reset,
  };

  return <InspectionContext.Provider value={value}>{children}</InspectionContext.Provider>;
}

export function useInspection() {
  const ctx = useContext(InspectionContext);
  if (!ctx) throw new Error("useInspection must be used within InspectionProvider");
  return ctx;
}
