import React, { useState, useEffect } from "react";
import {
  History,
  X,
  Clock,
  Trash2,
  RotateCcw,
  Sparkles,
  ShieldCheck,
  ImageIcon,
  LogIn,
  AlertCircle,
  Loader2,
} from "lucide-react";
import {
  getUserQueryHistory,
  deleteQueryHistoryItem,
  restoreHistoryImages,
} from "@/services/history";

export default function HistoryPanel({
  isOpen,
  onClose,
  user,
  supabase,
  onRestoreItem,
  openAuthModal,
}) {
  const [historyList, setHistoryList] = useState([]);
  const [loading, setLoading] = useState(false);
  const [restoringId, setRestoringId] = useState(null);
  const [deletingId, setDeletingId] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);

  const fetchHistory = async () => {
    if (!user) {
      setHistoryList([]);
      return;
    }
    setLoading(true);
    setErrorMsg(null);
    try {
      const data = await getUserQueryHistory({ supabase, user });
      setHistoryList(data);
    } catch (err) {
      console.error("Failed to fetch query history:", err);
      setErrorMsg("Failed to load query history.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (isOpen) {
      fetchHistory();
    }
  }, [isOpen, user]);

  if (!isOpen) return null;

  const handleRestore = async (item) => {
    setRestoringId(item.id);
    try {
      let restoredImages = [];
      if (item.images && item.images.length > 0) {
        restoredImages = await restoreHistoryImages({
          supabase,
          imageRecords: item.images,
        });
      }

      onRestoreItem(item, restoredImages);
      onClose();
    } catch (err) {
      console.error("Error restoring history item:", err);
    } finally {
      setRestoringId(null);
    }
  };

  const handleDelete = async (e, historyId) => {
    e.stopPropagation();
    setDeletingId(historyId);
    try {
      const success = await deleteQueryHistoryItem({
        supabase,
        user,
        historyId,
      });
      if (success) {
        setHistoryList((prev) => prev.filter((h) => h.id !== historyId));
      }
    } catch (err) {
      console.error("Error deleting history item:", err);
    } finally {
      setDeletingId(null);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex justify-end bg-black/70 backdrop-blur-sm transition-opacity animate-in fade-in duration-200"
      onClick={onClose}
    >
      <div
        className="relative flex h-full w-full max-w-lg flex-col border-l border-cyan-400/20 bg-[#070d18] text-white shadow-2xl transition-transform duration-300"
        onClick={(e) => e.stopPropagation()}
      >
        {/* DRAWER HEADER */}
        <div className="flex items-center justify-between border-b border-white/10 p-6 bg-[#030712]/80">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-cyan-400/30 bg-cyan-400/10 text-cyan-300 shadow-[0_0_15px_rgba(34,211,238,0.15)]">
              <History className="h-5 w-5" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-white">Query History</h2>
              <p className="text-xs text-slate-400">
                Persistent satellite analysis history
              </p>
            </div>
          </div>

          <button
            type="button"
            onClick={onClose}
            className="flex h-8 w-8 items-center justify-center rounded-lg border border-white/10 bg-white/5 text-slate-400 hover:bg-white/10 hover:text-white transition"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* DRAWER CONTENT */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {/* ANONYMOUS USER STATE */}
          {!user ? (
            <div className="flex flex-col items-center justify-center rounded-2xl border border-dashed border-cyan-400/20 bg-cyan-950/10 p-8 text-center mt-12">
              <div className="flex h-14 w-14 items-center justify-center rounded-2xl border border-cyan-400/30 bg-cyan-400/10 text-cyan-300 mb-4">
                <LogIn className="h-7 w-7" />
              </div>
              <h3 className="text-lg font-bold text-white">
                Authentication Required
              </h3>
              <p className="mt-2 text-xs text-slate-400 max-w-xs leading-relaxed">
                Please log in or create an account to view and restore your persistent query and satellite imagery history across sessions.
              </p>
              <button
                type="button"
                onClick={() => {
                  onClose();
                  openAuthModal && openAuthModal("login");
                }}
                className="mt-6 flex items-center gap-2 rounded-xl border border-cyan-400/40 bg-gradient-to-r from-cyan-500 to-blue-500 px-5 py-2.5 text-xs font-semibold text-white shadow-[0_0_20px_rgba(34,211,238,0.2)] hover:scale-105 transition"
              >
                <LogIn className="h-4 w-4" />
                <span>Log In / Sign Up</span>
              </button>
            </div>
          ) : loading ? (
            /* LOADING STATE */
            <div className="flex flex-col items-center justify-center py-20 text-center">
              <Loader2 className="h-8 w-8 animate-spin text-cyan-400 mb-3" />
              <p className="text-xs text-slate-400 font-mono">
                Retrieving stored user history...
              </p>
            </div>
          ) : historyList.length === 0 ? (
            /* EMPTY HISTORY STATE */
            <div className="flex flex-col items-center justify-center rounded-2xl border border-dashed border-white/10 bg-black/20 p-8 text-center mt-12">
              <Clock className="h-10 w-10 text-slate-600 mb-3" />
              <h3 className="text-base font-bold text-slate-300">
                No Query History Found
              </h3>
              <p className="mt-1 text-xs text-slate-500 max-w-xs">
                Your future satellite analysis queries and imagery payloads will be saved here automatically.
              </p>
            </div>
          ) : (
            /* HISTORY LIST ITEMS */
            historyList.map((item) => {
              const isRestoring = restoringId === item.id;
              const isDeleting = deletingId === item.id;
              const formattedDate = new Date(
                item.created_at
              ).toLocaleDateString("en-US", {
                month: "short",
                day: "numeric",
                year: "numeric",
                hour: "2-digit",
                minute: "2-digit",
              });

              return (
                <div
                  key={item.id}
                  className="group relative overflow-hidden rounded-2xl border border-white/10 bg-white/[0.02] p-5 transition-all duration-300 hover:border-cyan-400/40 hover:bg-white/[0.04]"
                >
                  {/* CARD TOP INFO */}
                  <div className="flex items-start justify-between gap-3 mb-2">
                    <div className="flex items-center gap-2">
                      <span className="rounded-md border border-cyan-400/30 bg-cyan-400/10 px-2 py-0.5 text-[10px] font-mono font-semibold text-cyan-300 uppercase">
                        {(item.task_detected || "Analysis").replace("_", " ")}
                      </span>
                      {item.confidence && (
                        <span className="rounded-md border border-emerald-400/30 bg-emerald-400/10 px-2 py-0.5 text-[10px] font-mono text-emerald-300">
                          {item.confidence}
                        </span>
                      )}
                    </div>

                    <div className="flex items-center gap-1.5 text-[11px] text-slate-400 font-mono">
                      <Clock className="h-3 w-3 text-slate-500" />
                      <span>{formattedDate}</span>
                    </div>
                  </div>

                  {/* QUERY TEXT */}
                  <h4 className="text-sm font-bold text-white line-clamp-2 mb-2">
                    "{item.query}"
                  </h4>

                  {/* ANSWER PREVIEW */}
                  {item.answer && (
                    <p className="text-xs text-slate-300 line-clamp-2 mb-3 bg-black/30 p-2.5 rounded-xl border border-white/5 font-sans">
                      {item.answer}
                    </p>
                  )}

                  {/* IMAGE THUMBNAILS ROW */}
                  {item.images && item.images.length > 0 && (
                    <div className="mb-4 flex items-center gap-2">
                      <div className="flex -space-x-2 overflow-hidden">
                        {item.images.slice(0, 3).map((img, idx) => (
                          <div
                            key={img.id || idx}
                            className="inline-block h-8 w-8 rounded-lg border border-cyan-400/30 bg-black/60 overflow-hidden shrink-0"
                          >
                            {img.public_url ? (
                              <img
                                src={img.public_url}
                                alt={img.file_name}
                                className="h-full w-full object-cover"
                              />
                            ) : (
                              <div className="flex h-full w-full items-center justify-center bg-cyan-950 text-cyan-400">
                                <ImageIcon className="h-3.5 w-3.5" />
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                      <span className="text-[11px] font-mono text-slate-400">
                        {item.images.length} asset
                        {item.images.length !== 1 ? "s" : ""}
                      </span>
                    </div>
                  )}

                  {/* ACTION BUTTONS FOOTER */}
                  <div className="flex items-center justify-between border-t border-white/10 pt-3 mt-1">
                    <span className="text-[10px] font-mono text-slate-500 truncate max-w-[180px]">
                      Model: {item.model_used || "SatQuery Controller"}
                    </span>

                    <div className="flex items-center gap-2">
                      {/* DELETE BUTTON */}
                      <button
                        type="button"
                        disabled={isDeleting}
                        onClick={(e) => handleDelete(e, item.id)}
                        className="flex h-8 w-8 items-center justify-center rounded-lg border border-red-500/20 bg-red-950/20 text-slate-400 hover:border-red-500/50 hover:bg-red-950/40 hover:text-red-300 transition"
                        title="Delete History Entry"
                      >
                        {isDeleting ? (
                          <Loader2 className="h-3.5 w-3.5 animate-spin text-red-400" />
                        ) : (
                          <Trash2 className="h-3 w-3" />
                        )}
                      </button>

                      {/* RESTORE BUTTON */}
                      <button
                        type="button"
                        disabled={isRestoring}
                        onClick={() => handleRestore(item)}
                        className="flex items-center gap-1.5 rounded-lg border border-cyan-400/40 bg-cyan-500/10 px-3 py-1.5 text-xs font-semibold text-cyan-300 hover:bg-cyan-500/20 hover:border-cyan-400 transition"
                      >
                        {isRestoring ? (
                          <>
                            <Loader2 className="h-3.5 w-3.5 animate-spin text-cyan-400" />
                            <span>Restoring...</span>
                          </>
                        ) : (
                          <>
                            <RotateCcw className="h-3.5 w-3.5 text-cyan-400" />
                            <span>Restore Session</span>
                          </>
                        )}
                      </button>
                    </div>
                  </div>
                </div>
              );
            })
          )}
        </div>

        {/* DRAWER FOOTER */}
        {user && historyList.length > 0 && (
          <div className="border-t border-white/10 bg-[#030712]/90 p-4 text-center text-[11px] font-mono text-slate-500">
            Clicking "Restore Session" populates prompt &amp; satellite images
          </div>
        )}
      </div>
    </div>
  );
}
