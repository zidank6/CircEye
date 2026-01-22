import React from 'react';
import type { CircuitInfo } from '../types';

interface ExplanationPanelProps {
    circuits: CircuitInfo[];
}

export function ExplanationPanel({ circuits }: ExplanationPanelProps) {
    return (
        <div className="explanation-panel">
            <h3>Insights</h3>

            <div className="explanation-section">
                <h4>What am I looking at?</h4>
                <p>
                    <strong>Attention</strong> shows how the model moves information between words.
                    Brighter cells mean the model is "focusing" more on that relationship.
                </p>
            </div>

            {circuits.length > 0 && (
                <div className="explanation-section">
                    <h4>Detected Patterns</h4>
                    <ul className="circuit-list">
                        {circuits.map((c, i) => (
                            <li key={i} className="circuit-item">
                                <div className="circuit-header">
                                    <span className="circuit-type">{c.type.replace('_', ' ')}</span>
                                    <span className="circuit-loc">L{c.layer}H{c.head}</span>
                                </div>
                                <p>{c.explanation}</p>
                            </li>
                        ))}
                    </ul>
                </div>
            )}

            <div className="explanation-section">
                <h4>Definitions</h4>
                <dl>
                    <dt>Induction Head</dt>
                    <dd>A circuit that completes patterns (e.g., knows "Harry" follows "Potter" if it saw it before).</dd>

                    <dt>Logit Lens</dt>
                    <dd>Decodes the model's internal state at each step to see what it "thinks" the next word is.</dd>
                </dl>
            </div>
        </div>
    );
}
