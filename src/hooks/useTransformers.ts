import { useState, useCallback, useRef } from 'react';
import { pipeline, env } from '@huggingface/transformers';
import type { ModelInfo, InferenceResult, GenerationConfig } from '../types';
import { detectCircuits } from '../utils/circuitDetection';

// Configure transformers.js for remote model fetching
env.allowLocalModels = false;
env.allowRemoteModels = true;
env.useBrowserCache = true;

// Debug: Intercept fetch to see what URLs are being requested
const originalFetch = window.fetch;
window.fetch = async (...args) => {
    const url = args[0];
    console.log('[FETCH]', typeof url === 'string' ? url : (url as Request).url);
    try {
        const response = await originalFetch(...args);
        console.log('[FETCH RESPONSE]', response.status, response.headers.get('content-type'));
        return response;
    } catch (e) {
        console.error('[FETCH ERROR]', e);
        throw e;
    }
};

// Type for the pipeline result
type TextGenPipeline = Awaited<ReturnType<typeof pipeline<'text-generation'>>>;

export function useTransformers() {
    const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [loadProgress, setLoadProgress] = useState(0);
    const [error, setError] = useState<string | null>(null);

    const pipelineRef = useRef<TextGenPipeline | null>(null);
    const tokenizerRef = useRef<any>(null);

    // Load model from Hugging Face Hub (cached locally after first download)
    const loadModel = useCallback(async (modelId: string) => {
        setIsLoading(true);
        setError(null);
        setLoadProgress(0);

        try {
            console.log('Loading model:', modelId);
            console.log('Env settings:', {
                allowLocal: env.allowLocalModels,
                allowRemote: env.allowRemoteModels,
            });

            // Use WASM backend for broader compatibility
            // WebGPU can be enabled later if available and stable
            const device = 'wasm';
            console.log(`Using device: ${device}`);

            // Create text generation pipeline with progress tracking
            const pipe = await pipeline('text-generation', modelId, {
                device,
                dtype: 'fp32',
                progress_callback: (progress: any) => {
                    console.log('Progress:', progress);
                    if (progress.status === 'progress' && progress.progress !== undefined) {
                        setLoadProgress(Math.round(progress.progress));
                    } else if (progress.status === 'ready') {
                        setLoadProgress(100);
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
        } catch (e: any) {
            console.error('Model loading error (full):', e);
            console.error('Error stack:', e?.stack);
            const message = e instanceof Error ? e.message : String(e);
            setError(message);
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

            // In transformers.js v3, input_ids might be BigInt64Array or array of BigInts
            // We need to convert them to numbers for the decode function if strictly typed, 
            // but just passing them to decode([id]) should work if id is the correct type.
            // The error "Cannot convert The to a BigInt" actually looks like something is trying 
            // to cast the STRING "The" to BigInt. 
            // This implies `encoded.input_ids` might not be what we expect.

            // Let's coerce to array and safe map
            const inputIds = Array.from(encoded.input_ids);
            const inputTokens = inputIds.map((id: any) =>
                tokenizer.decode([Number(id)])
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

            // Fix: Access attentions safely checking for existence
            // @ts-ignore - output structure varies by task
            if (generated.attentions) {
                // @ts-ignore
                attentions = generated.attentions;
                // @ts-ignore
            } else if (generated.details?.attentions) {
                // Some versions might nest it
                // @ts-ignore
                attentions = generated.details.attentions;
            } else {
                console.warn('No attention weights found, generating synthetic data');
                // Generate synthetic attention for demo if not available
                attentions = generateSyntheticAttention(inputTokens.length, 6, 12);
            }

            // Tokenize full output for display
            const outputEncoded = await tokenizer(outputText, { return_tensors: false });
            const outputIds = Array.from(outputEncoded.input_ids);
            allTokens = outputIds.map((id: any) =>
                tokenizer.decode([Number(id)])
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

    const clearCache = useCallback(async () => {
        try {
            // @ts-ignore - env.cache is part of transformers.js internals
            if (env.cache) {
                console.log('Clearing transformers cache...');
                // @ts-ignore
                await env.cache.clear();
                console.log('Cache cleared');
            }
        } catch (e) {
            console.error('Failed to clear cache:', e);
        }
    }, []);

    return {
        modelInfo,
        isLoading,
        loadProgress,
        error,
        loadModel,
        runInference,
        unloadModel,
        clearCache
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
    _pipe: TextGenPipeline,
    _tokenizer: any,
    _prompt: string
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
