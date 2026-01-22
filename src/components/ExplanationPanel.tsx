import type { CircuitInfo } from '../types';

interface ExplanationPanelProps {
    circuits: CircuitInfo[];
    onPromptSelect?: (prompt: string) => void;
}

const SUGGESTED_PROMPTS = [
    'The cat sat on the mat. The cat sat on the',
    'Once upon a time, once upon a',
    'AB CD AB',
    'Paris is to France as Tokyo is to',
];

export function ExplanationPanel({ circuits, onPromptSelect }: ExplanationPanelProps) {
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
                    <dd>
                        A circuit that completes patterns by copying what came after similar tokens.
                        <div className="test-prompt">
                            <strong>Test it:</strong> Try "The cat sat on the mat. The cat sat on the"
                            <br/>
                            Look for attention from final "the" → "mat"
                        </div>
                    </dd>

                    <dt>Previous Token Head</dt>
                    <dd>
                        Always attends to the immediately preceding token. Shows as a bright diagonal line.
                    </dd>

                    <dt>Duplicate Token Head</dt>
                    <dd>
                        Attends to repeated words in the sequence.
                        <div className="test-prompt">
                            <strong>Test it:</strong> Try "dog dog cat dog"
                            <br/>
                            Look for attention between the "dog" tokens
                        </div>
                    </dd>

                    <dt>Logit Lens</dt>
                    <dd>Decodes the model's internal state at each layer to see what it "thinks" the next word is.</dd>
                </dl>
            </div>

            <div className="explanation-section">
                <h4>Try These Prompts</h4>
                <p className="click-hint">Click to use:</p>
                <ul className="prompt-suggestions">
                    {SUGGESTED_PROMPTS.map((prompt, i) => (
                        <li key={i}>
                            <code onClick={() => onPromptSelect?.(prompt)}>
                                {prompt}
                            </code>
                        </li>
                    ))}
                </ul>
            </div>
        </div>
    );
}
