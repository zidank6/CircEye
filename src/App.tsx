import React, { useState } from 'react';
import { ModelLoader } from './components/ModelLoader';
import { PromptInput } from './components/PromptInput';
import { AttentionViz } from './components/AttentionViz';
import { LogitLens } from './components/LogitLens';
import { CircuitGraph } from './components/CircuitGraph';
import { AblationPanel } from './components/AblationPanel';
import { ExplanationPanel } from './components/ExplanationPanel';
import { useTransformers } from './hooks/useTransformers';
import { exportAsPng, exportAsSvg } from './utils/exportViz';
import type { GenerationConfig, InferenceResult } from './types';

function App() {
    const {
        modelInfo,
        isLoading,
        loadProgress,
        error,
        loadModel,
        runInference,
        unloadModel
    } = useTransformers();

    const [result, setResult] = useState<InferenceResult | null>(null);
    const [svgElement, setSvgElement] = useState<SVGSVGElement | null>(null);
    const [selectedPrompt, setSelectedPrompt] = useState<string | undefined>(undefined);

    const handleRun = async (prompt: string, config: GenerationConfig) => {
        const res = await runInference(prompt, config);
        setResult(res);
    };

    const handleExport = async (format: 'png' | 'svg') => {
        if (!svgElement) return;
        if (format === 'png') {
            await exportAsPng(svgElement);
        } else {
            await exportAsSvg(svgElement);
        }
    };

    return (
        <div className="app-container">
            <header className="app-header">
                <h1>Local LLM Circuit Visualizer</h1>
                <div className="status-badge">
                    {modelInfo?.loaded ?
                        <span className="status-online">Model Loaded: {modelInfo.name}</span> :
                        <span className="status-offline">No Model Loaded</span>
                    }
                </div>
            </header>

            <main className="main-grid">
                <aside className="sidebar-left">
                    <ModelLoader
                        modelInfo={modelInfo}
                        isLoading={isLoading}
                        loadProgress={loadProgress}
                        error={error}
                        onLoad={loadModel}
                        onUnload={unloadModel}
                    />

                    <div className="divider" />

                    <PromptInput
                        onRun={handleRun}
                        isRunning={isLoading}
                        disabled={!modelInfo?.loaded}
                        externalPrompt={selectedPrompt}
                    />

                    {modelInfo?.loaded && (
                        <>
                            <div className="divider" />
                            <AblationPanel
                                numLayers={modelInfo.numLayers}
                                numHeads={modelInfo.numHeads}
                                onAblate={(mask) => console.log('Ablation simulation only requires re-run logic', mask)}
                            />
                        </>
                    )}
                </aside>

                <section className="viz-center">
                    {result ? (
                        <>
                            <div className="viz-header">
                                <h2>Attention Patterns</h2>
                                <div className="export-controls">
                                    <button onClick={() => handleExport('png')}>Export PNG</button>
                                    <button onClick={() => handleExport('svg')}>Export SVG</button>
                                </div>
                            </div>

                            {/* Show generated output prominently */}
                            <div className="generated-output">
                                <h3>Model Output:</h3>
                                <div className="output-text">{result.output}</div>
                                <div className="output-note">
                                    Note: DistilGPT-2 is a small model (82M params). It may not handle complex reasoning well.
                                    Try simpler patterns like "The cat sat on the mat. The cat sat on the"
                                </div>
                            </div>

                            <div className="viz-container">
                                <AttentionViz
                                    attentions={result.attentions}
                                    attentionSource={result.attentionSource}
                                    tokens={result.tokens}
                                    numLayers={modelInfo?.numLayers || 6}
                                    numHeads={modelInfo?.numHeads || 12}
                                    onSvgRef={setSvgElement}
                                />
                            </div>

                            <div className="lens-container">
                                <LogitLens topPredictions={result.topPredictions} />
                            </div>
                        </>
                    ) : (
                        <div className="empty-state">
                            <p>Load a model and run a prompt to see visualizations.</p>
                        </div>
                    )}
                </section>

                <aside className="sidebar-right">
                    {result && (
                        <CircuitGraph circuits={result.circuits} />
                    )}
                    <ExplanationPanel
                        circuits={result?.circuits || []}
                        onPromptSelect={setSelectedPrompt}
                    />
                </aside>
            </main>
        </div>
    );
}

export default App;
