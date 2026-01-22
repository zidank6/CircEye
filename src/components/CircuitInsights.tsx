import type { DetectedCircuit } from '../utils/circuitDetection';
import { getCircuitTypeName, getCircuitColor, getConfidenceBadge } from '../utils/circuitDetection';

interface CircuitInsightsProps {
    circuits: DetectedCircuit[];
    onSelectCircuit?: (circuit: DetectedCircuit) => void;
    onAblateCircuit?: (circuit: DetectedCircuit) => void;
}

export function CircuitInsights({ circuits, onSelectCircuit, onAblateCircuit }: CircuitInsightsProps) {
    if (!circuits.length) {
        return (
            <div className="circuit-insights">
                <h3>Circuit Insights</h3>
                <p className="no-circuits">No interpretable circuits detected. Try prompts with repeated patterns like "The cat sat on the mat. The cat"</p>
            </div>
        );
    }

    // Group circuits by type
    const byType = circuits.reduce((acc, c) => {
        if (!acc[c.type]) acc[c.type] = [];
        acc[c.type].push(c);
        return acc;
    }, {} as Record<string, DetectedCircuit[]>);

    return (
        <div className="circuit-insights">
            <h3>Detected Circuits ({circuits.length})</h3>

            {Object.entries(byType).map(([type, typeCircuits]) => (
                <div key={type} className="circuit-type-group">
                    <div
                        className="circuit-type-header"
                        style={{ borderLeftColor: getCircuitColor(type as DetectedCircuit['type']) }}
                    >
                        <span className="type-name">{getCircuitTypeName(type as DetectedCircuit['type'])}</span>
                        <span className="type-count">{typeCircuits.length}</span>
                    </div>

                    <div className="circuit-list">
                        {typeCircuits.slice(0, 5).map((circuit, i) => {
                            const badge = getConfidenceBadge(circuit.confidence);
                            return (
                                <div
                                    key={i}
                                    className="circuit-item"
                                    onClick={() => onSelectCircuit?.(circuit)}
                                >
                                    <div className="circuit-header">
                                        <span className="circuit-location">
                                            L{circuit.layer}H{circuit.head}
                                        </span>
                                        <span
                                            className="confidence-badge"
                                            style={{ backgroundColor: badge.color + '30', color: badge.color }}
                                        >
                                            {circuit.confidence}
                                        </span>
                                        <span className="circuit-score">
                                            {(circuit.score * 100).toFixed(0)}%
                                        </span>
                                    </div>

                                    {circuit.evidence && (
                                        <div className="circuit-evidence">
                                            {circuit.evidence}
                                        </div>
                                    )}

                                    {onAblateCircuit && (
                                        <button
                                            className="ablate-btn"
                                            onClick={(e) => {
                                                e.stopPropagation();
                                                onAblateCircuit(circuit);
                                            }}
                                            title="Ablate this head to see its importance"
                                        >
                                            Test Ablation
                                        </button>
                                    )}
                                </div>
                            );
                        })}
                        {typeCircuits.length > 5 && (
                            <div className="more-circuits">
                                +{typeCircuits.length - 5} more
                            </div>
                        )}
                    </div>
                </div>
            ))}
        </div>
    );
}
