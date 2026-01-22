import { useState } from 'react';
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
    { id: 'Xenova/gpt2', name: 'GPT-2', size: '548MB', desc: 'Recommended', recommended: true },
    { id: 'Xenova/distilgpt2', name: 'DistilGPT-2', size: '353MB', desc: 'Faster', recommended: false },
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
            {modelInfo?.loaded ? (
                <div className="model-loaded">
                    <div className="loaded-info">
                        <span className="loaded-icon">●</span>
                        <span className="loaded-name">{modelInfo.name}</span>
                    </div>
                    <button onClick={onUnload} className="btn-small btn-ghost">
                        Unload
                    </button>
                </div>
            ) : (
                <>
                    <div className="model-grid">
                        {PRESET_MODELS.map(model => (
                            <button
                                key={model.id}
                                onClick={() => handlePresetLoad(model.id)}
                                disabled={isLoading}
                                className={`model-card ${model.recommended ? 'recommended' : ''}`}
                            >
                                <div className="model-card-header">
                                    <span className="model-card-name">{model.name}</span>
                                    <span className="model-card-size">{model.size}</span>
                                </div>
                                {model.recommended && <span className="recommended-badge">Recommended</span>}
                            </button>
                        ))}
                    </div>

                    <button
                        onClick={() => setShowCustom(!showCustom)}
                        className="btn-link"
                    >
                        {showCustom ? '− Hide custom' : '+ Custom model'}
                    </button>

                    {showCustom && (
                        <div className="custom-model">
                            <input
                                type="text"
                                value={customModelId}
                                onChange={e => setCustomModelId(e.target.value)}
                                placeholder="Xenova/model-name"
                                disabled={isLoading}
                            />
                            <button
                                onClick={handleCustomLoad}
                                disabled={isLoading || !customModelId.trim()}
                                className="btn-small btn-primary"
                            >
                                Load
                            </button>
                        </div>
                    )}
                </>
            )}

            {isLoading && (
                <div className="loading-container">
                    <div className="loading-bar">
                        <div
                            className="loading-progress"
                            style={{ width: `${loadProgress}%` }}
                        />
                    </div>
                    <span className="loading-text">{loadProgress}%</span>
                </div>
            )}

            {error && <div className="error-message">{error}</div>}
        </div>
    );
}
