import React from 'react';
import type { TokenPrediction } from '../types';

interface LogitLensProps {
    topPredictions: TokenPrediction[][];
}

export function LogitLens({ topPredictions }: LogitLensProps) {
    if (!topPredictions.length) return null;

    // Assume final layer is the output, show previous layers if available
    // For MVP we simulated 6 layers
    return (
        <div className="logit-lens">
            <h3>Logit Lens (Top-5 Predictions per Layer)</h3>
            <div className="lens-grid">
                {topPredictions.map((layerPreds, layerIdx) => (
                    <div key={layerIdx} className="lens-layer">
                        <h4>Layer {layerIdx}</h4>
                        <div className="lens-bars">
                            {layerPreds.map((pred, i) => (
                                <div key={i} className="lens-item">
                                    <div className="lens-token">'{pred.token}'</div>
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
                ))}
            </div>
        </div>
    );
}
