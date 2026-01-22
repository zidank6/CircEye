import { useState } from 'react';

interface ExplanationPanelProps {
    onPromptSelect?: (prompt: string) => void;
}

const SUGGESTED_PROMPTS = [
    { text: 'The cat sat on the mat. The cat sat on the', label: 'Induction' },
    { text: 'John gave Mary a book. Mary gave John a', label: 'Names' },
    { text: 'dog cat bird dog mouse dog', label: 'Duplicates' },
    { text: 'Paris is to France as Berlin is to', label: 'Analogy' },
    { text: 'one two three one two three one two', label: 'Pattern' },
    { text: 'The king had a crown. The queen had a crown. The king', label: 'Entity' },
];

interface CollapsibleProps {
    title: string;
    defaultOpen?: boolean;
    children: React.ReactNode;
}

function Collapsible({ title, defaultOpen = false, children }: CollapsibleProps) {
    const [isOpen, setIsOpen] = useState(defaultOpen);
    return (
        <div className="collapsible">
            <button className="collapsible-header" onClick={() => setIsOpen(!isOpen)}>
                <span>{title}</span>
                <span className="collapsible-icon">{isOpen ? '−' : '+'}</span>
            </button>
            {isOpen && <div className="collapsible-content">{children}</div>}
        </div>
    );
}

export function ExplanationPanel({ onPromptSelect }: ExplanationPanelProps) {
    return (
        <div className="explanation-panel">
            <div className="prompt-chips">
                <div className="chips-label">Try a prompt:</div>
                <div className="chips-grid">
                    {SUGGESTED_PROMPTS.map((prompt, i) => (
                        <button
                            key={i}
                            className="prompt-chip"
                            onClick={() => onPromptSelect?.(prompt.text)}
                            title={prompt.text}
                        >
                            {prompt.label}
                        </button>
                    ))}
                </div>
            </div>

            <Collapsible title="What am I looking at?" defaultOpen={false}>
                <p className="explanation-text">
                    The <strong>attention heatmap</strong> shows how the model moves information between tokens.
                    Brighter = stronger attention.
                </p>
            </Collapsible>

            <Collapsible title="Circuit Types" defaultOpen={false}>
                <div className="circuit-defs">
                    <div className="def-item">
                        <div className="def-name">Induction Head</div>
                        <div className="def-desc">Completes patterns: "A B ... A" → predicts "B"</div>
                    </div>
                    <div className="def-item">
                        <div className="def-name">Previous Token</div>
                        <div className="def-desc">Attends to prior token (diagonal pattern)</div>
                    </div>
                    <div className="def-item">
                        <div className="def-name">Duplicate Token</div>
                        <div className="def-desc">Links repeated words in sequence</div>
                    </div>
                </div>
            </Collapsible>

            <Collapsible title="Ablation Testing" defaultOpen={false}>
                <p className="explanation-text">
                    Select attention heads and "ablate" them to see how they affect the model's predictions.
                    Higher impact = more important head.
                </p>
            </Collapsible>
        </div>
    );
}
