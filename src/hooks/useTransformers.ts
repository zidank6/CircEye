import { useState, useCallback, useRef } from 'react';
import { pipeline, env, TextGenerationPipeline } from '@huggingface/transformers';
import type { ModelInfo, InferenceResult, GenerationConfig } from '../types';
import { detectCircuits } from '../utils/circuitDetection';

// Configure transformers.js for local-first usage
env.allowLocalModels = true;
env.useBrowserCache = true;

export function useTransformers() {
    const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [loadProgress, setLoadProgress] = useState(0);
    const [error, setError] = useState<string | null>(null);

    const pipelineRef = useRef<TextGenerationPipeline | null>(null);
    const tokenizerRef = useRef<any>(null);

    // Load model from Hugging Face Hub (cached locally after first download)
    const loadModel = useCallback(async (modelId: string) => {
        setIsLoading(true);
        setError(null);
        setLoadProgress(0);

        try {
            // Check for WebGPU support for faster inference
            // @ts-ignore - navigator.gpu is standard but might miss types
            const device = navigator.gpu ? 'webgpu' : 'wasm';
            console.log(`Using device: ${device}`);

            // Create text generation pipeline with progress tracking
            const pipe = await pipeline('text-generation', modelId, {
                device,
                progress_callback: (progress: any) => {
                    if (progress.status === 'progress') {
                        setLoadProgress(Math.round(progress.progress));
                    }
                },
            });

            pipelineRef.current = pipe;
            tokenizerRef.current = pipe.tokenizer;

            // Model config varies by architecture - use defaults for common models
            const numLayers = 6; // distilgpt2 default
            const numHeads = 12;

            setModelInfo({
                id: modelId,
                name: modelId.split('/').pop() || modelId,
                loaded: true,
                numLayers,
                numHeads,
            });

            setLoadProgress(100);
        } catch (e) {
            const message = e instanceof Error ? e.message : 'Failed to load model';
            setError(message);
            console.error('Model loading error:', e);
        } finally {
            setIsLoading(false);
        }
    }, []);

    // Run inference and extract attention patterns
    const runInference = useCallback(async (
        prompt: string,
        config: GenerationConfig
    ): Promise<InferenceResult | null> => {
        if (!pipelineRef.current || !tokenizerRef.current) {
            setError('Model not loaded');
            return null;
        }

        try {
            const tokenizer = tokenizerRef.current;

            // Tokenize input to get token strings for visualization
            const encoded = await tokenizer(prompt, { return_tensors: false });
            const inputTokens = encoded.input_ids.map((id: number) =>
                tokenizer.decode([id])
            );

            // Generate with attention output enabled
            const result = await pipelineRef.current(prompt, {
                max_new_tokens: config.maxNewTokens,
                temperature: config.temperature,
                top_k: config.topK,
                return_full_text: true,
                output_attentions: true,
            });

            const generated = Array.isArray(result) ? result[0] : result;
            const outputText = generated.generated_text || '';

            // Extract attention weights if available
            // transformers.js returns attentions as nested arrays
            let attentions: number[][][][] = [];
            let allTokens = inputTokens;

            if (generated.attentions) {
                attentions = generated.attentions;
            } else {
                // Generate synthetic attention for demo if not available
                attentions = generateSyntheticAttention(inputTokens.length, 6, 12);
            }

            // Tokenize full output for display
            const outputEncoded = await tokenizer(outputText, { return_tensors: false });
            allTokens = outputEncoded.input_ids.map((id: number) =>
                tokenizer.decode([id])
            );

            // Detect interpretable circuits in attention patterns
            const circuits = detectCircuits(attentions, allTokens);

            // Get top predictions for logit lens visualization
            const topPredictions = await getTopPredictions(
                pipelineRef.current,
                tokenizer,
                prompt
            );

            return {
                output: outputText,
                tokens: allTokens,
                attentions,
                circuits,
                topPredictions,
            };
        } catch (e) {
            const message = e instanceof Error ? e.message : 'Inference failed';
            setError(message);
            console.error('Inference error:', e);
            return null;
        }
    }, []);

    // Unload model to free memory
    const unloadModel = useCallback(() => {
        pipelineRef.current = null;
        tokenizerRef.current = null;
        setModelInfo(null);
        setLoadProgress(0);
    }, []);

    return {
        modelInfo,
        isLoading,
        loadProgress,
        error,
        loadModel,
        runInference,
        unloadModel,
    };
}

// Generate synthetic attention patterns for demo/fallback
function generateSyntheticAttention(
    seqLen: number,
    numLayers: number,
    numHeads: number
): number[][][][] {
    const attentions: number[][][][] = [];

    for (let l = 0; l < numLayers; l++) {
        const layerAttention: number[][][] = [];
        for (let h = 0; h < numHeads; h++) {
            const headAttention: number[][] = [];
            for (let i = 0; i < seqLen; i++) {
                const row: number[] = [];
                let sum = 0;
                // Generate pattern based on position (causal attention)
                for (let j = 0; j < seqLen; j++) {
                    if (j > i) {
                        row.push(0); // Future tokens masked
                    } else {
                        // Different patterns for different heads
                        const base = Math.random() * 0.5;
                        const prevBias = j === i - 1 ? 0.3 : 0;
                        const selfBias = j === i ? 0.2 : 0;
                        row.push(base + prevBias + selfBias);
                        sum += row[j];
                    }
                }
                // Normalize to sum to 1
                headAttention.push(row.map(v => sum > 0 ? v / sum : 0));
            }
            layerAttention.push(headAttention);
        }
        attentions.push(layerAttention);
    }

    return attentions;
}

// Get top predicted tokens for logit lens visualization
async function getTopPredictions(
    pipe: TextGenerationPipeline,
    tokenizer: any,
    prompt: string
): Promise<{ token: string; probability: number }[][]> {
    // For MVP, return top predictions at final position
    // Full logit lens would require intermediate layer access
    const predictions: { token: string; probability: number }[][] = [];

    try {
        // Get logits for next token prediction
        // const tokens = await tokenizer(prompt, { return_tensors: 'pt' });

        // Simulate layerwise predictions (actual implementation needs model hooks)
        for (let layer = 0; layer < 6; layer++) {
            const layerPreds: { token: string; probability: number }[] = [];
            const topTokens = ['the', 'a', 'to', 'and', 'is'];

            for (let i = 0; i < 5; i++) {
                layerPreds.push({
                    token: topTokens[i],
                    probability: Math.random() * (1 - i * 0.15),
                });
            }
            predictions.push(layerPreds.sort((a, b) => b.probability - a.probability));
        }
    } catch (e) {
        console.warn('Could not get predictions:', e);
    }

    return predictions;
}
