// Types for the circuit visualizer application

export interface ModelInfo {
    id: string;
    name: string;
    loaded: boolean;
    numLayers: number;
    numHeads: number;
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

export interface ExportOptions {
    format: 'png' | 'svg';
    width: number;
    height: number;
}
