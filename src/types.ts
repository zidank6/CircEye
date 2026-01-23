// Types for the circuit visualizer application

export interface ModelInfo {
    id: string;
    name: string;
    loaded: boolean;
    numLayers: number;
    numHeads: number;
    hiddenDim: number;
}

export interface AttentionData {
    layer: number;
    head: number;
    matrix: number[][]; // [seq_len, seq_len]
    tokens: string[];
}

export interface CircuitInfo {
    type: 'induction' | 'previous_token' | 'duplicate_token' | 'unknown';
    layer: number;
    head: number;
    score: number;
    explanation: string;
}

// Source of attention data - critical for research validity
export type AttentionSource = 'real' | 'kv_derived' | 'synthetic';

export interface InferenceResult {
    output: string;
    tokens: string[];
    attentions: number[][][][]; // [layers, heads, seq, seq]
    attentionSource: AttentionSource; // Indicates data quality for research
    circuits: CircuitInfo[];
    topPredictions: TokenPrediction[][];
    hiddenStates: number[][][] | null; // [layers+1, seq, hidden_dim] - includes embedding layer
    rawLogits: number[] | null; // Final position logits [vocab_size]
}

export interface AblationImpact {
    originalTopToken: string;
    originalTopProb: number;
    ablatedTopToken: string;
    ablatedTopProb: number;
    probabilityShift: number; // Change in top-1 probability
    entropyChange: number; // Change in output entropy (positive = more uncertain)
    klDivergence: number; // KL divergence between original and ablated distributions
    rankChanges: {
        token: string;
        originalRank: number;
        ablatedRank: number;
        probChange: number;
    }[];
}

export interface TokenPrediction {
    token: string;
    probability: number;
}

export interface GenerationConfig {
    maxNewTokens: number;
    temperature: number;
    topK: number;
}

export interface SteeringConfig {
    enabled: boolean;
    vector: Float32Array | null;
    strength: number;
    vectorName: string | null;
}

export interface ExportOptions {
    format: 'png' | 'svg';
    width: number;
    height: number;
}
