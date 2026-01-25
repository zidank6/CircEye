import type { TokenPrediction } from '../types';

interface LogitLensProps {
    topPredictions: TokenPrediction[][];
}

export function LogitLens({ topPredictions }: LogitLensProps) {
    if (!topPredictions || !topPredictions.length) {
        return (
            <div className="logit-lens">
                <h3>Next Token Predictions</h3>
                <div className="lens-item" style={{ opacity: 0.5, fontStyle: 'italic' }}>
                    No predictions available yet.
                </div>
            </div>
        );
    }

    // Show only the final layer predictions (actual next token probabilities)
    const finalPreds = topPredictions[topPredictions.length - 1];
    if (!finalPreds?.length) {
        return (
            <div className="logit-lens">
                <h3>Next Token Predictions</h3>
                <div className="lens-item" style={{ opacity: 0.5 }}>
                    Empty prediction set.
                </div>
            </div>
        );
    }

    return (
        <div className="logit-lens">
            <h3>Next Token Predictions (Top 5)</h3>
            <p className="lens-subtitle">What the model thinks comes next:</p>
            <div className="lens-bars">
                {finalPreds.map((pred, i) => (
                    <div key={i} className="lens-item">
                        <div className="lens-rank">#{i + 1}</div>
                        <div className="lens-token">"{pred.token}"</div>
                        <div className="lens-bar-container">
                            <div
                                className="lens-bar"
                                style={{ width: `${pred.probability * 100}%` }}
                            />
                        </div>
                        <div className="lens-prob">
                            {(pred.probability * 100).toFixed(1)}%
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}
