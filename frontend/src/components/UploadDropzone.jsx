import React, { useCallback, useRef, useState } from "react";

const ACCEPTED = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"];

export default function UploadDropzone({ file, previewUrl, onFile }) {
  const inputRef = useRef(null);
  const [dragOver, setDragOver] = useState(false);
  const [meta, setMeta] = useState(null);

  const handleFiles = useCallback(
    (files) => {
      const f = files?.[0];
      if (!f) return;
      const ext = "." + f.name.split(".").pop().toLowerCase();
      if (!ACCEPTED.includes(ext)) {
        alert(`Unsupported file type: ${ext}. Allowed: ${ACCEPTED.join(", ")}`);
        return;
      }
      const img = new Image();
      const url = URL.createObjectURL(f);
      img.onload = () => {
        setMeta({ width: img.width, height: img.height, sizeKb: (f.size / 1024).toFixed(1) });
      };
      img.src = url;
      onFile(f, url);
    },
    [onFile]
  );

  return (
    <div>
      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => { e.preventDefault(); setDragOver(false); handleFiles(e.dataTransfer.files); }}
        onClick={() => inputRef.current?.click()}
        className={`cursor-pointer border-2 border-dashed rounded-sm p-10 text-center transition-colors ${
          dragOver ? "border-accent-500 bg-accent-500/5" : "border-ink-600 hover:border-ink-500"
        }`}
      >
        {previewUrl ? (
          <div className="flex flex-col items-center gap-3">
            <img src={previewUrl} alt="preview" className="max-h-64 rounded-sm border border-ink-600" />
            <div className="text-xs font-mono text-slate-400">
              {file?.name} {meta && `· ${meta.width}×${meta.height}px · ${meta.sizeKb} KB`}
            </div>
            <button
              type="button"
              onClick={(e) => { e.stopPropagation(); inputRef.current?.click(); }}
              className="text-xs text-accent-400 hover:text-accent-300 font-medium"
            >
              Replace image
            </button>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-3">
            <div className="w-12 h-12 rounded-sm bg-ink-700 flex items-center justify-center">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
                <path d="M12 16V4M12 4L7 9M12 4l5 5" strokeLinecap="round" strokeLinejoin="round" />
                <path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </div>
            <div>
              <div className="font-medium text-slate-200">Drag & drop an image, or click to browse</div>
              <div className="text-xs text-slate-500 mt-1">Supported: JPG, PNG, BMP, TIFF · Max 15 MB</div>
            </div>
          </div>
        )}
      </div>
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED.join(",")}
        className="hidden"
        onChange={(e) => handleFiles(e.target.files)}
      />
    </div>
  );
}
