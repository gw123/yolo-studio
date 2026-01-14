import React, { useState, useEffect } from 'react';
import { X, GitMerge, ArrowRight } from 'lucide-react';
import { LabelClass } from '../types';

interface MergeModalProps {
    isOpen: boolean;
    onClose: () => void;
    labels: LabelClass[];
    onMerge: (sourceId: string, targetId: string) => void;
}

export const MergeModal: React.FC<MergeModalProps> = ({ isOpen, onClose, labels, onMerge }) => {
    const [sourceId, setSourceId] = useState<string>('');
    const [targetId, setTargetId] = useState<string>('');

    // Reset selection when modal opens
    useEffect(() => {
        if (isOpen && labels.length >= 2) {
            // Default to last merged into first, or any logic
            setSourceId(labels[labels.length - 1].id);
            setTargetId(labels[0].id);
        }
    }, [isOpen, labels]);

    if (!isOpen) return null;

    const handleConfirm = () => {
        if (sourceId && targetId && sourceId !== targetId) {
            onMerge(sourceId, targetId);
        }
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
            <div className="bg-neutral-900 border border-neutral-800 rounded-xl max-w-md w-full shadow-2xl overflow-hidden animate-in fade-in zoom-in duration-200">
                {/* Header */}
                <div className="flex items-center justify-between p-4 border-b border-neutral-800 bg-neutral-950">
                    <h3 className="font-semibold text-white flex items-center gap-2">
                        <GitMerge size={18} className="text-purple-500" /> Merge Classes
                    </h3>
                    <button onClick={onClose} className="text-neutral-500 hover:text-white transition-colors">
                        <X size={20} />
                    </button>
                </div>

                {/* Body */}
                <div className="p-6 space-y-6">
                    <p className="text-sm text-neutral-400">
                        Select two classes to merge. All annotations from the source class will be reassigned to the target class, and the <strong className="text-white">source class will be deleted</strong>.
                    </p>

                    <div className="flex flex-col gap-4 bg-neutral-950/50 p-4 rounded-lg border border-neutral-800/50">
                        <div className="space-y-2">
                            <label className="text-xs font-semibold text-neutral-500 uppercase flex items-center gap-2">
                                <span className="w-2 h-2 rounded-full bg-red-500"></span>
                                From (Source)
                            </label>
                            <select
                                value={sourceId}
                                onChange={e => setSourceId(e.target.value)}
                                className="w-full bg-neutral-900 border border-neutral-700 rounded-lg p-2.5 text-sm text-white focus:border-purple-500 outline-none transition-colors"
                            >
                                {labels.map(l => (
                                    <option key={l.id} value={l.id} disabled={l.id === targetId}>{l.name}</option>
                                ))}
                            </select>
                        </div>

                        <div className="flex justify-center text-neutral-600">
                            <ArrowRight size={20} />
                        </div>

                        <div className="space-y-2">
                            <label className="text-xs font-semibold text-neutral-500 uppercase flex items-center gap-2">
                                <span className="w-2 h-2 rounded-full bg-green-500"></span>
                                Into (Target)
                            </label>
                            <select
                                value={targetId}
                                onChange={e => setTargetId(e.target.value)}
                                className="w-full bg-neutral-900 border border-neutral-700 rounded-lg p-2.5 text-sm text-white focus:border-purple-500 outline-none transition-colors"
                            >
                                {labels.map(l => (
                                    <option key={l.id} value={l.id} disabled={l.id === sourceId}>{l.name}</option>
                                ))}
                            </select>
                        </div>
                    </div>

                    <div className="bg-yellow-900/10 border border-yellow-900/30 p-3 rounded flex gap-3 items-start text-yellow-500/90 text-xs">
                        <div className="mt-0.5">⚠️</div>
                        <div>
                            <strong>Warning:</strong> This action cannot be undone. Check your selection carefully before confirming.
                        </div>
                    </div>
                </div>

                {/* Footer */}
                <div className="p-4 border-t border-neutral-800 bg-neutral-950 flex justify-end gap-3">
                    <button onClick={onClose} className="px-4 py-2 rounded text-sm font-medium text-neutral-400 hover:text-white hover:bg-neutral-800 transition-colors">
                        Cancel
                    </button>
                    <button
                        onClick={handleConfirm}
                        disabled={!sourceId || !targetId || sourceId === targetId}
                        className="px-4 py-2 rounded text-sm font-medium bg-purple-600 hover:bg-purple-500 text-white disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-purple-900/20 transition-all active:scale-95"
                    >
                        Confirm Merge
                    </button>
                </div>
            </div>
        </div>
    );
};
