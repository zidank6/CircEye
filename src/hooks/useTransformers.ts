import { useState, useCallback, useRef } from 'react';
import { pipeline, env, Tensor } from '@huggingface/transformers';
import type { ModelInfo, InferenceResult, GenerationConfig, AttentionSource } from '../types';
import { detectCircuits } from '../utils/circuitDetection';

// Configure transformers.js for remote model fetching
env.allowLocalModels = false;
env.allowRemoteModels = true;
env.useBrowserCache = true;

// Type for the pipeline result
type TextGenPipeline = Awaited<ReturnType<typeof pipeline<'text-generation'>>>;

export function useTransformers() {
    const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [loadProgress, setLoadProgress] = useState(0);
    const [error, setError] = useState<string | null>(null);

    const pipelineRef = useRef<TextGenPipeline | null>(null);
    const tokenizerRef = useRef<any>(null);
    const modelRef = useRef<any>(null); // Raw model for attention extraction

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
            modelRef.current = pipe.model; // Access underlying model for attention extraction

            // Try to get actual model config for accurate layer/head counts
            const config = pipe.model?.config || {};
            const numLayers = config.n_layer || config.num_hidden_layers || 6;
            const numHeads = config.n_head || config.num_attention_heads || 12;

            console.log('Model config:', { numLayers, numHeads, config });

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

            // Generate text
            const result = await pipelineRef.current(prompt, {
                max_new_tokens: config.maxNewTokens,
                temperature: config.temperature,
                top_k: config.topK,
                return_full_text: true,
            });

            const generated = Array.isArray(result) ? result[0] : result;
            const outputText = generated.generated_text || '';

            // Get exact subword tokens from the model's tokenizer
            // This is critical for research accuracy - attention is computed at subword level
            const encoded = await tokenizer(outputText);
            let allTokens: string[] = [];

            // Access the tokenizer's vocabulary to convert IDs to token strings
            const inputIds = encoded.input_ids;

            // Get the underlying data as regular numbers
            let idsArray: bigint[] | number[];
            if (inputIds.tolist) {
                idsArray = inputIds.tolist().flat();
            } else if (inputIds.ort_tensor?.cpuData) {
                idsArray = Array.from(inputIds.ort_tensor.cpuData);
            } else if (inputIds.data) {
                idsArray = Array.from(inputIds.data);
            } else {
                idsArray = Array.from(inputIds);
            }

            console.log('Token IDs:', idsArray);

            // Decode tokens one by one using the tokenizer
            // In transformers.js v3, we need to create proper tensors for decode
            for (let i = 0; i < idsArray.length; i++) {
                const id = idsArray[i];
                try {
                    // Try different decode approaches
                    let tokenStr: string;

                    // Method 1: Use decode with array
                    if (tokenizer.decode_single) {
                        tokenStr = tokenizer.decode_single(Number(id));
                    } else {
                        // Method 2: Decode with the same tensor type as input
                        const singleId = inputIds.slice([i], [i + 1]);
                        tokenStr = tokenizer.decode(singleId, { skip_special_tokens: false });
                    }

                    // Clean up GPT-2 style space marker
                    tokenStr = tokenStr.replace(/^Ġ/, ' ').replace(/^▁/, ' ');
                    allTokens.push(tokenStr || `[${id}]`);
                } catch {
                    // Fallback: just use the ID
                    allTokens.push(`[${id}]`);
                }
            }

            // If all tokens failed, try batch decode and split
            if (allTokens.every(t => t.startsWith('['))) {
                console.warn('Individual decode failed, trying batch approach');
                const fullText = tokenizer.decode(inputIds, { skip_special_tokens: false });
                // Split on GPT-2 space markers or whitespace
                const splits = fullText.split(/(Ġ|▁|\s+)/).filter(Boolean);
                if (splits.length > 0) {
                    allTokens = splits.map(s => s.replace(/^Ġ/, ' ').replace(/^▁/, ' '));
                }
            }

            console.log('Exact subword tokens:', allTokens);

            // Extract REAL attention weights by calling model directly
            // This is critical for research - we need actual attention patterns
            let attentions: number[][][][] = [];
            let attentionSource: AttentionSource = 'synthetic';

            if (modelRef.current) {
                try {
                    console.log('Extracting real attention weights...');

                    // Create attention_mask (all 1s for non-padded input)
                    const seqLen = idsArray.length;
                    const attentionMaskData = new BigInt64Array(seqLen).fill(1n);
                    const attentionMask = new Tensor('int64', attentionMaskData, [1, seqLen]);

                    // Create position_ids (0, 1, 2, ..., seqLen-1)
                    const positionIdsData = new BigInt64Array(seqLen);
                    for (let i = 0; i < seqLen; i++) {
                        positionIdsData[i] = BigInt(i);
                    }
                    const positionIds = new Tensor('int64', positionIdsData, [1, seqLen]);

                    console.log('Calling model with attention_mask and position_ids...');

                    // Call model forward pass with all required inputs
                    const modelOutput = await modelRef.current({
                        input_ids: inputIds,
                        attention_mask: attentionMask,
                        position_ids: positionIds,
                        output_attentions: true,
                    });

                    console.log('Model output keys:', Object.keys(modelOutput));

                    if (modelOutput.attentions) {
                        // Convert attention tensors to nested arrays
                        attentions = await Promise.all(
                            modelOutput.attentions.map(async (layerAttn: any) => {
                                let data: Float32Array | number[];
                                if (layerAttn.data) {
                                    data = layerAttn.data;
                                } else if (layerAttn.ort_tensor?.cpuData) {
                                    data = layerAttn.ort_tensor.cpuData;
                                } else {
                                    data = await layerAttn.tolist();
                                }

                                const dims = layerAttn.dims || layerAttn.shape || [1, 12, allTokens.length, allTokens.length];
                                const [_batch, numHeads, attnSeqLen, _seqLen2] = dims;
                                const headsData: number[][][] = [];
                                const headSize = attnSeqLen * attnSeqLen;

                                for (let h = 0; h < numHeads; h++) {
                                    const headMatrix: number[][] = [];
                                    for (let i = 0; i < attnSeqLen; i++) {
                                        const row: number[] = [];
                                        for (let j = 0; j < attnSeqLen; j++) {
                                            const idx = h * headSize + i * attnSeqLen + j;
                                            row.push(Number(data[idx]) || 0);
                                        }
                                        headMatrix.push(row);
                                    }
                                    headsData.push(headMatrix);
                                }
                                return headsData;
                            })
                        );
                        attentionSource = 'real';
                        console.log(`✓ Extracted REAL attention: ${attentions.length} layers`);
                    } else {
                        // Try to compute attention from KV cache
                        console.warn('No attention field - attempting KV cache extraction');
                        const kvAttention = extractAttentionFromKV(modelOutput, seqLen);
                        if (kvAttention) {
                            attentions = kvAttention;
                            attentionSource = 'kv_derived';
                            console.log('✓ Computed attention from KV cache (K@K^T approximation)');
                        } else {
                            console.warn('⚠ Using SYNTHETIC attention - not research grade!');
                            attentions = generateSyntheticAttention(allTokens.length, 6, 12);
                            attentionSource = 'synthetic';
                        }
                    }
                } catch (attnError) {
                    console.error('Failed to extract attention:', attnError);
                    attentions = generateSyntheticAttention(allTokens.length, 6, 12);
                    attentionSource = 'synthetic';
                }
            } else {
                console.warn('No model reference, using synthetic attention');
                attentions = generateSyntheticAttention(allTokens.length, 6, 12);
                attentionSource = 'synthetic';
            }

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
                attentionSource,
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
        modelRef.current = null;
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

// Extract attention-like patterns from KV cache
// This computes K@K^T which shows key similarity - not true attention but informative
function extractAttentionFromKV(modelOutput: any, seqLen: number): number[][][][] | null {
    try {
        const attentions: number[][][][] = [];
        let layerIdx = 0;

        // Look for present.X.key tensors
        while (modelOutput[`present.${layerIdx}.key`]) {
            const keyTensor = modelOutput[`present.${layerIdx}.key`];

            // Get tensor data and dimensions
            const data = keyTensor.data || keyTensor.ort_tensor?.cpuData;
            const dims = keyTensor.dims || keyTensor.shape;

            if (!data || !dims) {
                layerIdx++;
                continue;
            }

            // dims typically: [batch, num_heads, seq_len, head_dim]
            // But can vary - log and handle gracefully
            console.log(`KV layer ${layerIdx} dims:`, dims);

            if (dims.length < 4) {
                console.warn(`Unexpected dims length: ${dims.length}, skipping layer`);
                layerIdx++;
                continue;
            }

            const [_batch, numHeads, tensorSeqLen, headDim] = dims;

            if (!numHeads || !tensorSeqLen || !headDim) {
                console.warn(`Invalid dims values: heads=${numHeads}, seq=${tensorSeqLen}, dim=${headDim}`);
                layerIdx++;
                continue;
            }

            const actualSeqLen = Math.min(seqLen, tensorSeqLen);

            const layerAttention: number[][][] = [];

            for (let h = 0; h < numHeads; h++) {
                const headMatrix: number[][] = [];

                // Compute K @ K^T for this head (key similarity matrix)
                for (let i = 0; i < actualSeqLen; i++) {
                    const row: number[] = [];
                    for (let j = 0; j < actualSeqLen; j++) {
                        // Dot product of key[i] and key[j]
                        let dotProduct = 0;
                        for (let d = 0; d < headDim; d++) {
                            const idx_i = h * tensorSeqLen * headDim + i * headDim + d;
                            const idx_j = h * tensorSeqLen * headDim + j * headDim + d;
                            dotProduct += Number(data[idx_i]) * Number(data[idx_j]);
                        }
                        // Apply causal mask
                        if (j > i) {
                            row.push(0);
                        } else {
                            row.push(dotProduct / Math.sqrt(headDim));
                        }
                    }
                    // Softmax normalize the row
                    const maxVal = Math.max(...row.filter((_, idx) => idx <= i));
                    const expRow = row.map((v, idx) => idx <= i ? Math.exp(v - maxVal) : 0);
                    const sumExp = expRow.reduce((a, b) => a + b, 0);
                    headMatrix.push(expRow.map(v => sumExp > 0 ? v / sumExp : 0));
                }
                layerAttention.push(headMatrix);
            }

            attentions.push(layerAttention);
            layerIdx++;
        }

        if (attentions.length > 0) {
            console.log(`Extracted ${attentions.length} layers from KV cache (K@K^T approximation)`);
            return attentions;
        }

        return null;
    } catch (e) {
        console.error('KV extraction failed:', e);
        return null;
    }
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
