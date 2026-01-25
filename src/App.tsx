import { useState, useEffect, useCallback } from 'react';
import './App.css';
import './steer.css';
import { ModelLoader } from './components/ModelLoader';
import { PromptInput } from './components/PromptInput';
import { AttentionViz } from './components/AttentionViz';
import { LogitLens } from './components/LogitLens';
import { CircuitGraph } from './components/CircuitGraph';
import { AblationPanel } from './components/AblationPanel';
import { AblationComparison } from './components/AblationComparison';
import { CircuitInsights } from './components/CircuitInsights';
import { ExplanationPanel } from './components/ExplanationPanel';
import { VectorLab } from './components/VectorLab';
import { SteeringPanel } from './components/SteeringPanel';
import { useTransformers } from './hooks/useTransformers';
import { useAblation } from './hooks/useAblation';
import { exportAsPng, exportAsSvg } from './utils/exportViz';
import type { GenerationConfig, InferenceResult, SteeringConfig } from './types';
import type { DetectedCircuit } from './utils/circuitDetection';
import type { AblationMask } from './hooks/useAblation';
import type { SteeringVectorData } from './utils/vectorMath';
import { loadSteeringVector } from './utils/vectorMath';

const STEERING_VECTORS_STORAGE_KEY = 'circeye_steering_vectors';

function App() {
    const {
        modelInfo,
        isLoading,
        loadProgress,
        error,
        loadModel,
        runInference,
        unloadModel,
        getTokenizer,
        getHiddenState
    } = useTransformers();

    const {
        ablationResult,
        isComparing,
        runAblation,
        clearAblation
    } = useAblation();

    const [result, setResult] = useState<InferenceResult | null>(null);
    const [svgElement, setSvgElement] = useState<SVGSVGElement | null>(null);
    const [selectedPrompt, setSelectedPrompt] = useState<string | undefined>(undefined);

    // Steering state
    const [savedVectors, setSavedVectors] = useState<SteeringVectorData[]>([]);
    const [activeVector, setActiveVector] = useState<SteeringVectorData | null>(null);
    const [steeringStrength, setSteeringStrength] = useState(0);
    const [steeringEnabled, setSteeringEnabled] = useState(true);

    // Load saved vectors from localStorage on mount
    useEffect(() => {
        try {
            const stored = localStorage.getItem(STEERING_VECTORS_STORAGE_KEY);
            if (stored) {
                const vectors = JSON.parse(stored) as SteeringVectorData[];
                setSavedVectors(vectors);
                console.log(`Loaded ${vectors.length} saved steering vectors`);
            }
        } catch (e) {
            console.warn('Failed to load saved vectors:', e);
        }
    }, []);

    // Save vectors to localStorage whenever they change
    useEffect(() => {
        try {
            localStorage.setItem(STEERING_VECTORS_STORAGE_KEY, JSON.stringify(savedVectors));
        } catch (e) {
            console.warn('Failed to save vectors:', e);
        }
    }, [savedVectors]);

    // Get current steering config
    const getSteeringConfig = useCallback((): SteeringConfig => {
        if (!activeVector || !steeringEnabled) {
            return {
                enabled: false,
                vector: null,
                strength: 0,
                vectorName: null,
            };
        }

        return {
            enabled: true,
            vector: loadSteeringVector(activeVector),
            strength: steeringStrength,
            vectorName: activeVector.name,
        };
    }, [activeVector, steeringEnabled, steeringStrength]);

    const handleRun = async (prompt: string, config: GenerationConfig) => {
        clearAblation(); // Clear any previous ablation when running new inference
        const steeringConfig = getSteeringConfig();
        const res = await runInference(prompt, config, steeringConfig);
        setResult(res);
    };

    // Handlers for Vector Lab
    const handleVectorCreated = useCallback((vector: SteeringVectorData) => {
        setSavedVectors(prev => {
            // Replace if name exists, otherwise add
            const existing = prev.findIndex(v => v.name === vector.name);
            if (existing >= 0) {
                const updated = [...prev];
                updated[existing] = vector;
                return updated;
            }
            return [...prev, vector];
        });
        // Auto-load the newly created vector
        setActiveVector(vector);
        setSteeringStrength(1); // Default strength
    }, []);

    const handleLoadVector = useCallback((vector: SteeringVectorData) => {
        setActiveVector(vector);
        setSteeringStrength(1);
        setSteeringEnabled(true);
    }, []);

    const handleDeleteVector = useCallback((id: string) => {
        setSavedVectors(prev => prev.filter(v => v.id !== id));
        if (activeVector?.id === id) {
            setActiveVector(null);
        }
    }, [activeVector]);

    const handleComputeHiddenStates = useCallback(async (text: string): Promise<Float32Array | null> => {
        return getHiddenState(text);
    }, [getHiddenState]);

    const handleExport = async (format: 'png' | 'svg') => {
        if (!svgElement) return;
        if (format === 'png') {
            await exportAsPng(svgElement);
        } else {
            await exportAsSvg(svgElement);
        }
    };

    const handleAblation = (mask: AblationMask[]) => {
        if (result?.attentions) {
            runAblation(result.attentions, mask, result.rawLogits, getTokenizer());
        }
    };

    const handleAblateCircuit = (circuit: DetectedCircuit) => {
        if (result?.attentions) {
            runAblation(result.attentions, [{ layer: circuit.layer, head: circuit.head }], result.rawLogits, getTokenizer());
        }
    };

    // Cast circuits to DetectedCircuit (they have the extended properties from detection)
    const detectedCircuits = (result?.circuits || []) as DetectedCircuit[];

    // Use ablated attentions if available, otherwise original
    const displayAttentions = isComparing && ablationResult?.ablatedAttentions
        ? ablationResult.ablatedAttentions
        : result?.attentions || null;

    return (
        <div className="app-container">
            <header className="app-header">
                <h1>Local LLM Circuit Visualizer</h1>
                <div className="header-badges">
                    {activeVector && steeringEnabled && (
                        <span className="steering-badge">
                            STEERING: {activeVector.name} ({steeringStrength > 0 ? '+' : ''}{steeringStrength.toFixed(1)})
                        </span>
                    )}
                    <div className="status-badge">
                        {modelInfo?.loaded ?
                            <span className="status-online">Model Loaded: {modelInfo.name}</span> :
                            <span className="status-offline">No Model Loaded</span>
                        }
                    </div>
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
                                detectedCircuits={detectedCircuits}
                                onAblate={handleAblation}
                                onClear={clearAblation}
                            />

                            <div className="divider" />
                            <SteeringPanel
                                config={getSteeringConfig()}
                                onChange={(cfg) => {
                                    setSteeringEnabled(cfg.enabled);
                                    setSteeringStrength(cfg.strength);
                                }}
                                activeVectorData={activeVector}
                            />

                            <div className="divider" />
                            <VectorLab
                                onCompute={handleComputeHiddenStates}
                                onVectorCreated={handleVectorCreated}
                                savedVectors={savedVectors}
                                onLoadVector={handleLoadVector}
                                onDeleteVector={handleDeleteVector}
                            />
                        </>
                    )}
                </aside>

                <section className="viz-center">
                    {result ? (
                        <>
                            <div className="viz-header">
                                <h2>
                                    Attention Patterns
                                    {isComparing && <span className="ablation-badge">ABLATED VIEW</span>}
                                </h2>
                                <div className="export-controls">
                                    <button onClick={() => handleExport('png')}>Export PNG</button>
                                    <button onClick={() => handleExport('svg')}>Export SVG</button>
                                </div>
                            </div>

                            {/* Show ablation comparison if active */}
                            {isComparing && ablationResult && (
                                <AblationComparison
                                    result={ablationResult}
                                    onClose={clearAblation}
                                />
                            )}

                            {/* Show generated output prominently */}
                            <div className="generated-output">
                                <h3>Model Output:</h3>
                                <div className="output-text">{result.output}</div>
                                <div className="output-note">
                                    {modelInfo?.name.includes('Qwen') ? (
                                        <span>
                                            <strong>Qwen 1.5 (0.5B)</strong> is a capable base model. It works best with <strong>completions</strong> (e.g., code snippets, story continuation) and <strong>few-shot examples</strong>.
                                            It is not an instruction-tuned chat model, so it won't "answer" questions directly unless prodded.
                                        </span>
                                    ) : (
                                        <span>
                                            Note: Small models like DistilGPT-2 (82M) struggle with complex logic.
                                            Try simple repetitions or pattern completions.
                                        </span>
                                    )}
                                </div>
                            </div>

                            <div className="viz-container">
                                <AttentionViz
                                    attentions={displayAttentions}
                                    attentionSource={isComparing ? 'synthetic' : result.attentionSource}
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
                        <>
                            <CircuitInsights
                                circuits={detectedCircuits}
                                onAblateCircuit={handleAblateCircuit}
                            />
                            <div className="divider" />
                            <CircuitGraph circuits={result.circuits} />
                        </>
                    )}
                    <ExplanationPanel
                        onPromptSelect={setSelectedPrompt}
                    />
                </aside>
            </main>
        </div>
    );
}

export default App;
