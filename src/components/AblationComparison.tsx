import type { AblationResult } from '../hooks/useAblation';

interface AblationComparisonProps {
    result: AblationResult;
    onClose: () => void;
}

export function AblationComparison({ result, onClose }: AblationComparisonProps) {
    const { ablatedHeads, impactScore } = result;

    // Determine impact severity
    const getImpactLevel = (score: number): { label: string; color: string; description: string } => {
        if (score > 0.3) return {
            label: 'High Impact',
            color: '#e74c3c',
            description: 'These heads significantly affect attention patterns'
        };
        if (score > 0.15) return {
            label: 'Medium Impact',
            color: '#f39c12',
            description: 'These heads have moderate influence on attention'
        };
        return {
            label: 'Low Impact',
            color: '#27ae60',
            description: 'These heads have minimal effect on attention patterns'
        };
    };

    const impact = getImpactLevel(impactScore);

    return (
        <div className="ablation-comparison">
            <div className="ablation-header">
                <h3>Ablation Analysis</h3>
                <button className="close-btn" onClick={onClose}>×</button>
            </div>

            <div className="ablated-heads">
                <span className="label">Ablated:</span>
                {ablatedHeads.map((h, i) => (
                    <span key={i} className="head-tag">
                        L{h.layer}H{h.head}
                    </span>
                ))}
            </div>

            <div
                className="impact-indicator"
                style={{ backgroundColor: impact.color + '20', borderColor: impact.color }}
            >
                <div className="impact-header">
                    <span className="impact-label" style={{ color: impact.color }}>
                        {impact.label}
                    </span>
                    <span className="impact-score">
                        {(impactScore * 100).toFixed(1)}% change
                    </span>
                </div>
                <p className="impact-description">{impact.description}</p>
            </div>

            <div className="impact-bar-container">
                <div className="impact-bar-label">Attention Pattern Change</div>
                <div className="impact-bar-bg">
                    <div
                        className="impact-bar-fill"
                        style={{
                            width: `${Math.min(100, impactScore * 100)}%`,
                            backgroundColor: impact.color
                        }}
                    />
                </div>
            </div>

            <div className="ablation-note">
                <strong>Note:</strong> This is a simulation that zeroes out the specified attention heads.
                The impact score shows how much the overall attention pattern changed.
                Real ablation would require model modification and re-inference.
            </div>
        </div>
    );
}
