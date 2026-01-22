import type { AblationResult } from '../hooks/useAblation';

interface AblationComparisonProps {
    result: AblationResult;
    onClose: () => void;
}

export function AblationComparison({ result, onClose }: AblationComparisonProps) {
    const { ablatedHeads, impactScore, realImpact, headContributions } = result;

    // Determine impact severity based on real impact if available
    const getImpactLevel = (score: number, realImpact: typeof result.realImpact): {
        label: string;
        color: string;
        description: string;
    } => {
        // Use KL divergence or probability shift for real impact assessment
        const effectiveScore = realImpact
            ? Math.min(1, realImpact.klDivergence * 5 + Math.abs(realImpact.probabilityShift) * 2)
            : score;

        if (effectiveScore > 0.3) return {
            label: 'High Impact',
            color: '#e74c3c',
            description: realImpact
                ? 'These heads significantly affect model predictions'
                : 'These heads significantly affect attention patterns'
        };
        if (effectiveScore > 0.15) return {
            label: 'Medium Impact',
            color: '#f39c12',
            description: realImpact
                ? 'These heads have moderate influence on predictions'
                : 'These heads have moderate influence on attention'
        };
        return {
            label: 'Low Impact',
            color: '#27ae60',
            description: realImpact
                ? 'These heads have minimal effect on predictions'
                : 'These heads have minimal effect on attention patterns'
        };
    };

    const impact = getImpactLevel(impactScore, realImpact);

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

            {/* Real Impact Section - Only shown when we have actual output analysis */}
            {realImpact && (
                <div className="real-impact-section">
                    <div className="prediction-comparison">
                        <div className="prediction-box original">
                            <span className="pred-label">Original Prediction</span>
                            <span className="pred-token">"{realImpact.originalTopToken}"</span>
                            <span className="pred-prob">{(realImpact.originalTopProb * 100).toFixed(1)}%</span>
                        </div>
                        <div className="prediction-arrow">→</div>
                        <div className="prediction-box ablated">
                            <span className="pred-label">After Ablation</span>
                            <span className="pred-token">"{realImpact.ablatedTopToken}"</span>
                            <span className="pred-prob">{(realImpact.ablatedTopProb * 100).toFixed(1)}%</span>
                        </div>
                    </div>

                    <div className="real-metrics">
                        <div className="metric-row">
                            <span className="metric-label">Confidence Change</span>
                            <span className={`metric-value ${realImpact.probabilityShift < 0 ? 'negative' : 'positive'}`}>
                                {realImpact.probabilityShift > 0 ? '+' : ''}{(realImpact.probabilityShift * 100).toFixed(1)}%
                            </span>
                        </div>
                        <div className="metric-row">
                            <span className="metric-label">Entropy Change</span>
                            <span className={`metric-value ${realImpact.entropyChange > 0 ? 'warning' : 'positive'}`}>
                                {realImpact.entropyChange > 0 ? '+' : ''}{realImpact.entropyChange.toFixed(3)} bits
                            </span>
                        </div>
                        <div className="metric-row">
                            <span className="metric-label">KL Divergence</span>
                            <span className="metric-value">{realImpact.klDivergence.toFixed(4)}</span>
                        </div>
                    </div>

                    {/* Rank Changes */}
                    {realImpact.rankChanges.length > 0 && (
                        <div className="rank-changes">
                            <div className="rank-header">Top Token Rank Changes</div>
                            {realImpact.rankChanges.slice(0, 5).map((change, i) => (
                                <div key={i} className="rank-row">
                                    <span className="rank-token">"{change.token}"</span>
                                    <span className="rank-positions">
                                        #{change.originalRank} → #{change.ablatedRank}
                                    </span>
                                    <span className={`rank-prob-change ${change.probChange > 0 ? 'positive' : 'negative'}`}>
                                        {change.probChange > 0 ? '+' : ''}{(change.probChange * 100).toFixed(1)}%
                                    </span>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            )}

            {/* Head Contributions */}
            {headContributions.length > 0 && (
                <div className="head-contributions">
                    <div className="contributions-header">Head Importance Ranking</div>
                    {headContributions.slice(0, 5).map((hc, i) => (
                        <div key={i} className="contribution-row">
                            <span className="contribution-head">L{hc.layer}H{hc.head}</span>
                            <div className="contribution-bar-container">
                                <div
                                    className="contribution-bar"
                                    style={{ width: `${Math.min(100, hc.percentage)}%` }}
                                />
                            </div>
                            <span className="contribution-pct">{hc.percentage.toFixed(1)}%</span>
                        </div>
                    ))}
                </div>
            )}

            <div
                className="impact-indicator"
                style={{ backgroundColor: impact.color + '20', borderColor: impact.color }}
            >
                <div className="impact-header">
                    <span className="impact-label" style={{ color: impact.color }}>
                        {impact.label}
                    </span>
                    <span className="impact-score">
                        {realImpact
                            ? `KL: ${realImpact.klDivergence.toFixed(3)}`
                            : `${(impactScore * 100).toFixed(1)}% attn change`
                        }
                    </span>
                </div>
                <p className="impact-description">{impact.description}</p>
            </div>

            <div className="impact-bar-container">
                <div className="impact-bar-label">
                    {realImpact ? 'Output Distribution Change' : 'Attention Pattern Change'}
                </div>
                <div className="impact-bar-bg">
                    <div
                        className="impact-bar-fill"
                        style={{
                            width: `${Math.min(100, realImpact
                                ? (realImpact.klDivergence * 500 + Math.abs(realImpact.probabilityShift) * 200)
                                : impactScore * 100
                            )}%`,
                            backgroundColor: impact.color
                        }}
                    />
                </div>
            </div>

            <div className="ablation-note">
                <strong>Note:</strong> {realImpact
                    ? 'This analysis estimates output changes based on attention head importance. Actual model behavior may differ due to non-linear interactions between heads.'
                    : 'Enable raw logits extraction to see predicted output changes. Currently showing attention pattern differences only.'
                }
            </div>
        </div>
    );
}
