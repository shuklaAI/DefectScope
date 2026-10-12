import React from "react";
import { Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar.jsx";
import { InspectionProvider } from "./context/InspectionContext.jsx";

import Home from "./pages/Home.jsx";
import Inspection from "./pages/Inspection.jsx";
import Pipeline from "./pages/Pipeline.jsx";
import Results from "./pages/Results.jsx";
import History from "./pages/History.jsx";
import Dashboard from "./pages/Dashboard.jsx";
import About from "./pages/About.jsx";

export default function App() {
  return (
    <InspectionProvider>
      <div className="min-h-screen flex flex-col">
        <Navbar />
        <main className="flex-1">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/inspection" element={<Inspection />} />
            <Route path="/pipeline" element={<Pipeline />} />
            <Route path="/results" element={<Results />} />
            <Route path="/history" element={<History />} />
            <Route path="/history/:id" element={<Results />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/about" element={<About />} />
          </Routes>
        </main>
        <footer className="border-t border-ink-600 py-6 mt-16">
          <div className="max-w-7xl mx-auto px-5 text-xs text-slate-500 flex flex-wrap gap-2 justify-between">
            <span>DefectScope | Academic digital image processing project</span>
            <span className="font-mono">FastAPI · OpenCV · scikit-image · SciPy · scikit-learn · React</span>
          </div>
        </footer>
      </div>
    </InspectionProvider>
  );
}
