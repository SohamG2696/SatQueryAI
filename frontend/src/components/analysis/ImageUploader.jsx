import React, { useRef, useState } from "react";
import {
  UploadCloud,
  X,
  CheckCircle2,
  ImageIcon,
  AlertCircle,
} from "lucide-react";

const MIN_IMAGES = 1;
const ACCEPTED_TYPES = /\.(jpg|jpeg|png|webp|tif|tiff)$/i;

function fileToPayload(file) {
  return {
    isCustom: true,
    file,
    id: `${file.name}-${file.lastModified}-${file.size}`,
    name: file.name,
    ext: file.name.split(".").pop().toUpperCase(),
    size: (file.size / (1024 * 1024)).toFixed(2) + " MB",
    url: URL.createObjectURL(file),
    sensor: "Optical / SAR Sensor",
    resolution: "0.5m GSD",
    coordinates: "Auto-georeferenced",
    captureDate: new Date().toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
    }),
  };
}

function isSupported(file) {
  return (
    file.type.startsWith("image/") || ACCEPTED_TYPES.test(file.name)
  );
}

function ImageUploader({ images = [], onImagesChange }) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const addFiles = (fileList) => {
    const incoming = Array.from(fileList).filter(isSupported);
    if (!incoming.length) return;

    // Deduplicate by id
    const existingIds = new Set(images.map((img) => img.id));
    const newPayloads = incoming
      .map(fileToPayload)
      .filter((p) => !existingIds.has(p.id));

    if (newPayloads.length) {
      onImagesChange([...images, ...newPayloads]);
    }
  };

  const removeImage = (id) => {
    const updated = images.filter((img) => img.id !== id);
    onImagesChange(updated);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files?.length) addFiles(e.dataTransfer.files);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => setIsDragging(false);

  const count = images.length;
  const isReady = count >= MIN_IMAGES;
  const counterLabel = isReady
    ? `${count} image${count !== 1 ? "s" : ""} ready for analysis`
    : `${count} / ${MIN_IMAGES} images uploaded`;

  return (
    <div className="space-y-4">
      {/* DROP ZONE */}
      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={() => fileInputRef.current?.click()}
        className={`relative flex min-h-[180px] cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed p-6 transition-all duration-300 ${
          isDragging
            ? "border-cyan-400 bg-cyan-400/10 scale-[0.99] shadow-[0_0_30px_rgba(34,211,238,0.2)]"
            : "border-cyan-400/30 bg-cyan-400/[0.02] hover:border-cyan-400/60 hover:bg-cyan-400/[0.06]"
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept="image/jpeg,image/jpg,image/png,image/webp,image/tiff,.jpg,.jpeg,.png,.webp,.tif,.tiff"
          className="hidden"
          onChange={(e) => {
            if (e.target.files?.length) {
              addFiles(e.target.files);
              // reset input so same file can be re-added after removal
              e.target.value = "";
            }
          }}
        />

        <div className="flex h-14 w-14 items-center justify-center rounded-2xl border border-cyan-400/30 bg-cyan-400/10 text-cyan-300 shadow-[0_0_20px_rgba(34,211,238,0.15)] mb-3">
          <UploadCloud className="h-7 w-7" />
        </div>

        <p className="text-base font-semibold text-white">
          Upload Satellite Imagery
        </p>

        <p className="mt-1 text-xs text-slate-400 text-center max-w-xs">
          Upload 1 or more satellite images for VQA, grounding, or change detection.
        </p>

        <p className="mt-1 text-[11px] text-slate-500 text-center">
          Supports JPG · JPEG · PNG · WebP · GeoTIFF / TIF
        </p>

        <div className="mt-4 flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-4 py-1.5 text-xs text-slate-300 hover:bg-white/10 transition">
          <span>Browse Local Files</span>
          <span className="text-cyan-400">→</span>
        </div>

        <p className="mt-2 text-[10px] text-slate-500 font-mono">
          1+ images supported · Drag &amp; drop supported
        </p>
      </div>

      {/* IMAGE COUNT BADGE */}
      {count > 0 && (
        <div
          className={`flex items-center gap-2 rounded-xl border px-4 py-2.5 text-xs font-semibold transition-colors ${
            isReady
              ? "border-emerald-400/40 bg-emerald-950/30 text-emerald-300"
              : "border-amber-400/40 bg-amber-950/20 text-amber-300"
          }`}
        >
          {isReady ? (
            <CheckCircle2 className="h-4 w-4 text-emerald-400 shrink-0" />
          ) : (
            <AlertCircle className="h-4 w-4 text-amber-400 shrink-0" />
          )}
          <span>{counterLabel}</span>
          {!isReady && (
            <span className="ml-auto text-[10px] font-normal text-amber-400/70">
              {MIN_IMAGES - count} more needed
            </span>
          )}
        </div>
      )}

      {/* THUMBNAIL GRID */}
      {count > 0 && (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
          {images.map((img) => (
            <div
              key={img.id}
              className="group relative overflow-hidden rounded-xl border border-white/10 bg-black/40"
            >
              {/* THUMBNAIL */}
              <div className="aspect-video w-full overflow-hidden bg-black/60">
                <img
                  src={img.url}
                  alt={img.name}
                  className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-105"
                />
              </div>

              {/* REMOVE BUTTON */}
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  removeImage(img.id);
                }}
                title="Remove image"
                className="absolute top-1.5 right-1.5 flex h-6 w-6 items-center justify-center rounded-lg border border-white/20 bg-black/70 text-slate-300 backdrop-blur opacity-0 group-hover:opacity-100 transition-all hover:bg-red-950/80 hover:border-red-500/50 hover:text-red-300"
              >
                <X className="h-3 w-3" />
              </button>

              {/* FILE INFO FOOTER */}
              <div className="flex items-center gap-1.5 border-t border-white/10 bg-[#080e1a] px-2 py-1.5">
                <ImageIcon className="h-3 w-3 text-cyan-400 shrink-0" />
                <span
                  className="truncate text-[10px] font-mono text-slate-300"
                  title={img.name}
                >
                  {img.name}
                </span>
                <span className="ml-auto shrink-0 rounded bg-cyan-950/60 px-1 py-0.5 text-[9px] font-bold text-cyan-400 border border-cyan-400/20">
                  {img.ext}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default ImageUploader;
