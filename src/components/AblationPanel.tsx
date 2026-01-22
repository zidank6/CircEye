import { useState } from 'react';
import type { AblationMask } from '../hooks/useAblation';
import type { DetectedCircuit } from '../utils/circuitDetection';
import { getCircuitColor } from '../utils/circuitDetection';

interface AblationPanelProps {
    numLayers: number;
    numHeads: number;
    detectedCircuits?: DetectedCircuit[];
    onAblate: (mask: AblationMask[]) => void;
    onClear: () => void;
}

export function AblationPanel({
    numLayers,
    numHeads,
    detectedCircuits = [],
    onAblate,
    onClear
}: AblationPanelProps) {
    // Store selected heads as set of "L{layer}H{head}" strings
    const [selectedHeads, setSelectedHeads] = useState<Set<string>>(new Set());

    // Build a map of detected circuit locations for highlighting
    const circuitMap = new Map<string, DetectedCircuit>();
    detectedCircuits.forEach(c => {
        const key = `L${c.layer}H${c.head}`;
        // Keep highest scoring circuit for this head
        if (!circuitMap.has(key) || (circuitMap.get(key)?.score || 0) < c.score) {
            circuitMap.set(key, c);
        }
    });

    const toggleHead = (layer: number, head: number) => {
        const key = `L${layer}H${head}`;
        const newSet = new Set(selectedHeads);
        if (newSet.has(key)) {
            newSet.delete(key);
        } else {
            newSet.add(key);
        }
        setSelectedHeads(newSet);
    };

    const handleApply = () => {
        const mask: AblationMask[] = [];
        selectedHeads.forEach(key => {
            const match = key.match(/L(\d+)H(\d+)/);
            if (match) {
                mask.push({
                    layer: Number(match[1]),
                    head: Number(match[2])
                });
            }
        });
        onAblate(mask);
    };

    const handleClear = () => {
        setSelectedHeads(new Set());
        onClear();
    };

    const selectAllCircuits = () => {
        const newSet = new Set(selectedHeads);
        detectedCircuits.forEach(c => {
            newSet.add(`L${c.layer}H${c.head}`);
        });
        setSelectedHeads(newSet);
    };

    const selectCircuitType = (type: string) => {
        const newSet = new Set(selectedHeads);
        detectedCircuits.filter(c => c.type === type).forEach(c => {
            newSet.add(`L${c.layer}H${c.head}`);
        });
        setSelectedHeads(newSet);
    };

    // Get unique circuit types present
    const circuitTypes = [...new Set(detectedCircuits.map(c => c.type))];

    return (
        <div className="ablation-panel">
            <h3>Ablation Testing</h3>
            <p className="hint">Select attention heads to ablate (zero out) and see their impact.</p>

            {detectedCircuits.length > 0 && (
                <div className="quick-select">
                    <span className="quick-label">Quick select:</span>
                    <button
                        className="quick-btn"
                        onClick={selectAllCircuits}
                        title="Select all detected circuits"
                    >
                        All Circuits
                    </button>
                    {circuitTypes.map(type => (
                        <button
                            key={type}
                            className="quick-btn"
                            onClick={() => selectCircuitType(type)}
                            style={{ borderColor: getCircuitColor(type as DetectedCircuit['type']) }}
                        >
                            {type.replace('_', ' ')}
                        </button>
                    ))}
                </div>
            )}

            <div className="ablation-grid">
                <div className="grid-header">
                    <span className="layer-label"></span>
                    {Array.from({ length: Math.min(numHeads, 12) }, (_, h) => (
                        <span key={h} className="head-label">H{h}</span>
                    ))}
                </div>
                {Array.from({ length: numLayers }, (_, l) => (
                    <div key={l} className="ablation-row">
                        <span className="layer-label">L{l}</span>
                        {Array.from({ length: Math.min(numHeads, 12) }, (_, h) => {
                            const key = `L${l}H${h}`;
                            const isSelected = selectedHeads.has(key);
                            const circuit = circuitMap.get(key);
                            const circuitColor = circuit ? getCircuitColor(circuit.type) : undefined;

                            return (
                                <button
                                    key={h}
                                    className={`head-toggle ${isSelected ? 'selected' : ''} ${circuit ? 'has-circuit' : ''}`}
                                    onClick={() => toggleHead(l, h)}
                                    title={circuit
                                        ? `${circuit.type} (${(circuit.score * 100).toFixed(0)}%)`
                                        : `Toggle L${l}H${h}`
                                    }
                                    style={circuit ? {
                                        '--circuit-color': circuitColor
                                    } as React.CSSProperties : undefined}
                                >
                                    {circuit && <span className="circuit-dot" style={{ backgroundColor: circuitColor }} />}
                                </button>
                            );
                        })}
                    </div>
                ))}
            </div>

            {numHeads > 12 && (
                <p className="grid-note">Showing first 12 of {numHeads} heads</p>
            )}

            <div className="ablation-actions">
                <button
                    onClick={handleApply}
                    className="btn-primary"
                    disabled={selectedHeads.size === 0}
                >
                    Run Ablation ({selectedHeads.size} heads)
                </button>
                <button
                    onClick={handleClear}
                    className="btn-secondary"
                    disabled={selectedHeads.size === 0}
                >
                    Clear
                </button>
            </div>
        </div>
    );
}
