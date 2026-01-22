import React, { useState } from 'react';
import type { ModelInfo } from '../types';

interface ModelLoaderProps {
    modelInfo: ModelInfo | null;
    isLoading: boolean;
    loadProgress: number;
    error: string | null;
    onLoad: (modelId: string) => void;
    onUnload: () => void;
}

// Preset models known to work well with transformers.js
const PRESET_MODELS = [
    { id: 'Xenova/distilgpt2', name: 'DistilGPT-2 (50MB)', desc: 'Fast, small' },
    { id: 'Xenova/gpt2', name: 'GPT-2 (120MB)', desc: 'Standard size' },
    { id: 'Xenova/TinyStories-1M', name: 'TinyStories (5MB)', desc: 'Tiny, fast' },
];

export function ModelLoader({
    modelInfo,
    isLoading,
    loadProgress,
    error,
    onLoad,
    onUnload,
}: ModelLoaderProps) {
    const [customModelId, setCustomModelId] = useState('');
    const [showCustom, setShowCustom] = useState(false);

    const handlePresetLoad = (modelId: string) => {
        onLoad(modelId);
    };

    const handleCustomLoad = () => {
        if (customModelId.trim()) {
            onLoad(customModelId.trim());
        }
    };

    return (
        <div className="model-loader">
            <h3>Model</h3>

            {modelInfo?.loaded ? (
                <div className="model-loaded">
                    <span className="model-name">{modelInfo.name}</span>
                    <button onClick={onUnload} className="btn-secondary">
                        Unload
                    </button>
                </div>
            ) : (
                <>
                    <div className="preset-models">
                        {PRESET_MODELS.map(model => (
                            <button
                                key={model.id}
                                onClick={() => handlePresetLoad(model.id)}
                                disabled={isLoading}
                                className="model-btn"
                            >
                                <span className="model-btn-name">{model.name}</span>
                                <span className="model-btn-desc">{model.desc}</span>
                            </button>
                        ))}
                    </div>

                    <button
                        onClick={() => setShowCustom(!showCustom)}
                        className="btn-link"
                    >
                        {showCustom ? 'Hide custom' : 'Use custom model ID'}
                    </button>

                    {showCustom && (
                        <div className="custom-model">
                            <input
                                type="text"
                                value={customModelId}
                                onChange={e => setCustomModelId(e.target.value)}
                                placeholder="e.g., Xenova/distilgpt2"
                                disabled={isLoading}
                            />
                            <button
                                onClick={handleCustomLoad}
                                disabled={isLoading || !customModelId.trim()}
                                className="btn-primary"
                            >
                                Load
                            </button>
                        </div>
                    )}
                </>
            )}

            {isLoading && (
                <div className="loading-bar">
                    <div
                        className="loading-progress"
                        style={{ width: `${loadProgress}%` }}
                    />
                    <span>{loadProgress}%</span>
                </div>
            )}

            {error && <div className="error-message">{error}</div>}
        </div>
    );
}
