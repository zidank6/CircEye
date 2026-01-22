import React, { useState } from 'react';

interface AblationPanelProps {
    numLayers: number;
    numHeads: number;
    onAblate: (mask: boolean[][]) => void;
}

export function AblationPanel({ numLayers, numHeads, onAblate }: AblationPanelProps) {
    // Store mask as set of "L{layer}H{head}" strings for easier state management
    const [ablatedHeads, setAblatedHeads] = useState<Set<string>>(new Set());

    const toggleHead = (layer: number, head: number) => {
        const key = `L${layer}H${head}`;
        const newSet = new Set(ablatedHeads);
        if (newSet.has(key)) {
            newSet.delete(key);
        } else {
            newSet.add(key);
        }
        setAblatedHeads(newSet);
    };

    const handleApply = () => {
        // Convert set to boolean matrix
        const mask: boolean[][] = Array(numLayers).fill(false).map(() => Array(numHeads).fill(false));
        ablatedHeads.forEach(key => {
            const match = key.match(/L(\d+)H(\d+)/);
            if (match) {
                mask[Number(match[1])][Number(match[2])] = true;
            }
        });
        onAblate(mask);
    };

    return (
        <div className="ablation-panel">
            <h3>Ablation Simulation</h3>
            <p className="hint">Select heads to "turn off" and see impact.</p>

            <div className="ablation-grid">
                {Array.from({ length: numLayers }, (_, l) => (
                    <div key={l} className="ablation-row">
                        <span className="row-label">L{l}</span>
                        {Array.from({ length: numHeads }, (_, h) => {
                            const key = `L${l}H${h}`;
                            const isAblated = ablatedHeads.has(key);
                            return (
                                <button
                                    key={h}
                                    className={`head-toggle ${isAblated ? 'ablated' : ''}`}
                                    onClick={() => toggleHead(l, h)}
                                    title={`Toggle Head ${l}-${h}`}
                                >
                                    {h}
                                </button>
                            );
                        })}
                    </div>
                ))}
            </div>

            <button onClick={handleApply} className="btn-warning" disabled={ablatedHeads.size === 0}>
                Run Ablated (Simulation)
            </button>
        </div>
    );
}
