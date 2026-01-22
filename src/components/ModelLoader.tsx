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
    { id: 'onnx-community/Llama-3.2-1B-Instruct-q4f16', name: 'Llama-3.2-1B (q4)', desc: 'Small Llama' },
    { id: 'Xenova/distilgpt2', name: 'DistilGPT-2 (50MB)', desc: 'Fast, small' },
    { id: 'Xenova/gpt2', name: 'GPT-2 (120MB)', desc: 'Standard size' },
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
    const [testResult, setTestResult] = useState<string | null>(null);

    // Test network connectivity to Hugging Face
    const testNetwork = async () => {
        setTestResult('Testing...');
        try {
            const url = 'https://huggingface.co/Xenova/distilgpt2/resolve/main/config.json';
            console.log('Testing fetch to:', url);
            const response = await fetch(url);
            const contentType = response.headers.get('content-type');
            console.log('Response status:', response.status, 'Content-Type:', contentType);

            if (!response.ok) {
                setTestResult(`HTTP ${response.status}: ${response.statusText}`);
                return;
            }

            const text = await response.text();
            console.log('Response preview:', text.substring(0, 200));

            if (text.startsWith('<')) {
                setTestResult(`Got HTML instead of JSON: ${text.substring(0, 100)}`);
            } else {
                try {
                    JSON.parse(text);
                    setTestResult('✓ Network OK - JSON parsed successfully');
                } catch {
                    setTestResult(`Parse error: ${text.substring(0, 100)}`);
                }
            }
        } catch (e: any) {
            setTestResult(`Network error: ${e.message}`);
            console.error('Network test error:', e);
        }
    };

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

            <div style={{ marginTop: '12px' }}>
                <button onClick={testNetwork} className="btn-secondary" style={{ fontSize: '12px' }}>
                    Test Network
                </button>
                {testResult && (
                    <div style={{ marginTop: '8px', fontSize: '12px', color: testResult.startsWith('✓') ? '#4ecdc4' : '#ff6b6b' }}>
                        {testResult}
                    </div>
                )}
            </div>
        </div>
    );
}
