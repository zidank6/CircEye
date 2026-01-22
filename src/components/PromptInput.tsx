import React, { useState, useEffect } from 'react';
import type { GenerationConfig } from '../types';

interface PromptInputProps {
    onRun: (prompt: string, config: GenerationConfig) => void;
    isRunning: boolean;
    disabled: boolean;
    externalPrompt?: string; // Allow external control of prompt
}

export function PromptInput({ onRun, isRunning, disabled, externalPrompt }: PromptInputProps) {
    const [prompt, setPrompt] = useState('The quick brown fox');

    // Update prompt when external prompt changes
    useEffect(() => {
        if (externalPrompt !== undefined) {
            setPrompt(externalPrompt);
        }
    }, [externalPrompt]);
    const [maxTokens, setMaxTokens] = useState(20);
    const [temperature, setTemperature] = useState(0.7);

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (prompt.trim() && !disabled) {
            onRun(prompt, {
                maxNewTokens: maxTokens,
                temperature,
                topK: 50,
            });
        }
    };

    return (
        <form onSubmit={handleSubmit} className="prompt-input">
            <div className="prompt-main">
                <textarea
                    value={prompt}
                    onChange={e => setPrompt(e.target.value)}
                    placeholder="Enter your prompt..."
                    disabled={disabled || isRunning}
                    rows={3}
                />
                <button
                    type="submit"
                    disabled={disabled || isRunning || !prompt.trim()}
                    className="btn-primary run-btn"
                >
                    {isRunning ? 'Running...' : 'Run'}
                </button>
            </div>

            <div className="prompt-settings">
                <label>
                    Max tokens:
                    <input
                        type="number"
                        value={maxTokens}
                        onChange={e => setMaxTokens(Number(e.target.value))}
                        min={1}
                        max={100}
                        disabled={disabled || isRunning}
                    />
                </label>
                <label>
                    Temperature:
                    <input
                        type="number"
                        value={temperature}
                        onChange={e => setTemperature(Number(e.target.value))}
                        min={0}
                        max={2}
                        step={0.1}
                        disabled={disabled || isRunning}
                    />
                </label>
            </div>
        </form>
    );
}
