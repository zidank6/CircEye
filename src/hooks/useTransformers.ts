import { useState, useCallback, useRef } from 'react';
import { pipeline, env, Tensor } from '@huggingface/transformers';
import type { ModelInfo, InferenceResult, GenerationConfig, AttentionSource, SteeringConfig } from '../types';
import { detectCircuits } from '../utils/circuitDetection';
import { scale } from '../utils/vectorMath';

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
    const embeddingMatrixRef = useRef<Float32Array | null>(null); // Cached embedding matrix
    const modelIdRef = useRef<string | null>(null); // Track current model ID persistently

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

            // Use WebGPU if available, fallback to WASM
            // @ts-ignore - navigator.gpu
            const isQwen = modelId.toLowerCase().includes('qwen');
            // Force WASM for Qwen due to WebGPU GQA bug causing garbage output
            const device = isQwen ? 'wasm' : ((navigator as any).gpu ? 'webgpu' : 'wasm');
            console.log(`Using device: ${device}`);

            // Create text generation pipeline with progress tracking
            // Create text generation pipeline with progress tracking
            const pipe = await pipeline('text-generation', modelId, {
                device,
                dtype: isQwen ? 'q8' : 'fp32', // Use q8 for quantized Qwen, fp32 for others
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
            modelIdRef.current = modelId; // Store ID for inference callbacks

            // Try to get actual model config for accurate layer/head counts
            const config: any = pipe.model?.config || {};
            const numLayers = config.n_layer || config.num_hidden_layers || 24; // Qwen 0.5B has 24 layers
            const numHeads = config.n_head || config.num_attention_heads || 16; // Qwen 0.5B has 16 heads
            const hiddenDim = config.n_embd || config.hidden_size || 1024; // Qwen 0.5B has 1024 dim

            console.log('Model config:', { numLayers, numHeads, hiddenDim, config });

            // Try to extract and cache the embedding matrix for steering
            try {
                await extractEmbeddingMatrix(pipe.model, hiddenDim);
            } catch (embErr) {
                console.warn('Could not extract embedding matrix:', embErr);
            }

            setModelInfo({
                id: modelId,
                name: modelId.split('/').pop() || modelId,
                loaded: true,
                numLayers,
                numHeads,
                hiddenDim,
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

    // Extract embedding matrix from model for steering vector injection
    const extractEmbeddingMatrix = async (model: any, expectedHiddenDim: number) => {
        try {
            // Try to access word embeddings from different model architectures
            const embeddings = model?.model?.embed_tokens?.weight ||
                model?.transformer?.wte?.weight ||
                model?.embeddings?.word_embeddings?.weight;

            if (embeddings?.data) {
                embeddingMatrixRef.current = new Float32Array(embeddings.data);
                const shape = embeddings.dims || embeddings.shape;
                const actualHiddenDim = shape?.[1] || expectedHiddenDim;
                console.log('✓ Extracted embedding matrix:', {
                    shape,
                    size: embeddingMatrixRef.current.length,
                    hiddenDim: actualHiddenDim,
                });
            } else {
                console.warn('Could not find embedding weights in model');
            }
        } catch (e) {
            console.warn('Failed to extract embedding matrix:', e);
        }
    };

    // Get hidden state for a single text (for steering vector computation)
    const getHiddenState = useCallback(async (
        text: string
    ): Promise<Float32Array | null> => {
        if (!modelRef.current || !tokenizerRef.current) {
            console.error('getHiddenState: Model or tokenizer not loaded');
            return null;
        }

        try {
            console.log('getHiddenState: Processing text:', text);
            const tokenizer = tokenizerRef.current;
            const model = modelRef.current;
            const encoded = await tokenizer(text);
            const inputIds = encoded.input_ids;

            // Get sequence length
            let idsArray: (bigint | number)[];
            if (inputIds.tolist) {
                idsArray = inputIds.tolist().flat();
            } else if ((inputIds as any).ort_tensor?.cpuData) {
                idsArray = Array.from((inputIds as any).ort_tensor.cpuData);
            } else if ((inputIds as any).data) {
                idsArray = Array.from((inputIds as any).data);
            } else {
                idsArray = Array.from(inputIds as any);
            }

            const seqLen = idsArray.length;
            console.log('getHiddenState: Sequence length:', seqLen);

            // Create attention mask and position ids
            const attentionMaskData = new BigInt64Array(seqLen).fill(1n);
            const attentionMask = new Tensor('int64', attentionMaskData, [1, seqLen]);

            const positionIdsData = new BigInt64Array(seqLen);
            for (let i = 0; i < seqLen; i++) {
                positionIdsData[i] = BigInt(i);
            }
            const positionIds = new Tensor('int64', positionIdsData, [1, seqLen]);

            // Run model forward pass with hidden states output
            console.log('getHiddenState: Running model forward pass...');
            const modelOutput = await model({
                input_ids: inputIds,
                attention_mask: attentionMask,
                position_ids: positionIds,
                output_hidden_states: true,
            });

            console.log('getHiddenState: Model output keys:', Object.keys(modelOutput));

            // Method 1: Try to get hidden_states directly
            if (modelOutput?.hidden_states && modelOutput.hidden_states.length > 0) {
                console.log('getHiddenState: Found hidden_states, layers:', modelOutput.hidden_states.length);
                const lastLayer = modelOutput.hidden_states[modelOutput.hidden_states.length - 1];
                let data: Float32Array | number[];
                if (lastLayer.data) {
                    data = lastLayer.data;
                } else if (lastLayer.ort_tensor?.cpuData) {
                    data = lastLayer.ort_tensor.cpuData;
                } else {
                    console.warn('getHiddenState: Could not extract data from hidden state tensor');
                    return null;
                }

                const dims = lastLayer.dims || lastLayer.shape || [1, seqLen, 768];
                const hiddenDim = dims[dims.length - 1];
                console.log('getHiddenState: Hidden dim:', hiddenDim);

                // Get the last token's hidden state
                const lastTokenStart = (seqLen - 1) * hiddenDim;
                const result = new Float32Array(
                    Array.from(data.slice(lastTokenStart, lastTokenStart + hiddenDim)).map(Number)
                );
                console.log('getHiddenState: Extracted vector of length:', result.length);
                return result;
            }

            // Method 2: Fallback - use logits to approximate representation
            // This is less accurate but works when hidden_states aren't available
            if (modelOutput?.logits) {
                console.log('getHiddenState: Falling back to logits-based approximation');
                let logitsData: Float32Array | number[];
                if (modelOutput.logits.data) {
                    logitsData = modelOutput.logits.data;
                } else if (modelOutput.logits.ort_tensor?.cpuData) {
                    logitsData = modelOutput.logits.ort_tensor.cpuData;
                } else {
                    console.warn('getHiddenState: Could not extract logits data');
                    return null;
                }

                const logitsDims = modelOutput.logits.dims || modelOutput.logits.shape;
                const vocabSize = logitsDims[logitsDims.length - 1];

                // Use the last position's logits as a representation
                // This is a poor substitute but allows the feature to work
                const lastPosStart = (seqLen - 1) * vocabSize;

                // Take top-k logit values as a compressed representation
                // This gives us a fixed-size vector that captures the model's "state"
                const k = 768; // Match GPT-2's hidden dim
                const lastLogits = Array.from(logitsData.slice(lastPosStart, lastPosStart + vocabSize)).map(Number);

                // Sort by magnitude and take top-k indices and values
                const indexed = lastLogits.map((v, i) => ({ v: Math.abs(v), i, orig: v }));
                indexed.sort((a, b) => b.v - a.v);

                const result = new Float32Array(k);
                for (let i = 0; i < k && i < indexed.length; i++) {
                    result[i] = indexed[i].orig;
                }

                console.log('getHiddenState: Created logits-based vector of length:', result.length);
                return result;
            }

            console.error('getHiddenState: No hidden_states or logits in model output');
            return null;
        } catch (e) {
            console.error('getHiddenState: Error:', e);
            return null;
        }
    }, []);

    // Run inference and extract attention patterns
    const runInference = useCallback(async (
        prompt: string,
        config: GenerationConfig,
        steeringConfig?: SteeringConfig
    ): Promise<InferenceResult | null> => {
        if (!pipelineRef.current || !tokenizerRef.current) {
            setError('Model not loaded');
            return null;
        }

        try {
            const tokenizer = tokenizerRef.current;
            const shouldSteer = steeringConfig?.enabled &&
                steeringConfig.vector &&
                steeringConfig.strength !== 0;

            let result: any;

            if (shouldSteer && modelRef.current && steeringConfig?.vector) {
                // Steered inference: inject steering vector into embeddings
                console.log('Running steered inference with strength:', steeringConfig.strength);
                result = await runSteeredInference(
                    prompt,
                    config,
                    steeringConfig.vector,
                    steeringConfig.strength
                );
            } else {
                // Normal inference
                result = await pipelineRef.current(prompt, {
                    max_new_tokens: config.maxNewTokens,
                    temperature: config.temperature,
                    top_k: config.topK,
                    return_full_text: true,
                });
            }

            const generated: any = Array.isArray(result) ? result[0] : result;
            const outputText = generated.generated_text || '';

            // Get exact subword tokens from the model's tokenizer
            // This is critical for research accuracy - attention is computed at subword level
            const encoded = await tokenizer(outputText);
            let allTokens: string[] = [];

            // Access the tokenizer's vocabulary to convert IDs to token strings
            const inputIds = encoded.input_ids;

            // Get the underlying data as regular numbers
            let idsArray: (bigint | number)[];
            if (inputIds.tolist) {
                idsArray = inputIds.tolist().flat();
            } else if ((inputIds as any).ort_tensor?.cpuData) {
                idsArray = Array.from((inputIds as any).ort_tensor.cpuData);
            } else if ((inputIds as any).data) {
                idsArray = Array.from((inputIds as any).data);
            } else {
                idsArray = Array.from(inputIds as any);
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
                    allTokens = splits.map((s: string) => s.replace(/^Ġ/, ' ').replace(/^▁/, ' '));
                }
            }

            console.log('Exact subword tokens:', allTokens);

            // Extract REAL attention weights by calling model directly
            // This is critical for research - we need actual attention patterns
            let attentions: number[][][][] = [];
            let attentionSource: AttentionSource = 'synthetic';
            let modelOutput: any = null;

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
                    modelOutput = await modelRef.current({
                        input_ids: inputIds,
                        attention_mask: attentionMask,
                        position_ids: positionIds,
                        output_attentions: true,
                        output_hidden_states: true,
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
                        const kvResult = extractAttentionFromKV(modelOutput, seqLen);
                        if (kvResult) {
                            attentions = kvResult.attentions;
                            // Use 'real' if we computed Q@K^T, 'kv_derived' for K@K^T approximation
                            attentionSource = kvResult.isQKT ? 'real' : 'kv_derived';
                            console.log(`✓ Computed attention from KV cache (${kvResult.isQKT ? 'Q@K^T' : 'K@K^T'})`);
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
            // Detect interpretable circuits in attention patterns
            const circuits = detectCircuits(attentions, allTokens);

            // FORCE CORRECT VOCAB SIZE
            // Standard config is often wrong for quantized ONNX
            let vocabSize = 0;
            const currentModelId = modelIdRef.current || '';
            if (currentModelId.toLowerCase().includes('qwen')) {
                vocabSize = 151936; // Qwen 1.5 standard
            } else if (currentModelId.toLowerCase().includes('llama')) {
                vocabSize = 32000;
            } else {
                vocabSize = modelRef.current?.config?.vocab_size || 50257;
            }
            console.log('Forced inference vocab size:', vocabSize);

            // Get top predictions from model logits
            const numLayersForLens = attentions.length || 6;
            const topPredictions = await getTopPredictions(
                modelOutput,
                tokenizer,
                numLayersForLens,
                vocabSize // Pass explicit size
            );

            // Extract hidden states for real ablation computation
            let hiddenStates: number[][][] | null = null;
            if (modelOutput?.hidden_states) {
                try {
                    hiddenStates = await extractHiddenStates(modelOutput.hidden_states);
                    console.log(`✓ Extracted hidden states: ${hiddenStates.length} layers`);
                } catch (hsError) {
                    console.warn('Failed to extract hidden states:', hsError);
                }
            }

            // Extract raw logits for ablation impact computation
            let rawLogits: number[] | null = null;
            if (modelOutput?.logits) {
                try {
                    rawLogits = extractRawLogits(modelOutput.logits, vocabSize); // Pass explicit size
                    console.log(`✓ Extracted raw logits: ${rawLogits.length} entries`);
                } catch (logitError) {
                    console.warn('Failed to extract raw logits:', logitError);
                }
            }

            return {
                output: outputText,
                tokens: allTokens,
                attentions,
                attentionSource,
                circuits,
                topPredictions,
                hiddenStates,
                rawLogits,
            };
        } catch (e) {
            const message = e instanceof Error ? e.message : 'Inference failed';
            setError(message);
            console.error('Inference error:', e);
            return null;
        }
    }, []);

    // Run inference with steering vector injection into input embeddings
    const runSteeredInference = async (
        prompt: string,
        config: GenerationConfig,
        steeringVector: Float32Array,
        strength: number
    ): Promise<any> => {
        const tokenizer = tokenizerRef.current;
        const model = modelRef.current;

        if (!tokenizer || !model) {
            throw new Error('Model or tokenizer not available');
        }

        // Tokenize input
        const encoded = await tokenizer(prompt);
        const inputIds = encoded.input_ids;

        // Get sequence length
        let idsArray: (bigint | number)[];
        if (inputIds.tolist) {
            idsArray = inputIds.tolist().flat();
        } else if ((inputIds as any).ort_tensor?.cpuData) {
            idsArray = Array.from((inputIds as any).ort_tensor.cpuData);
        } else if ((inputIds as any).data) {
            idsArray = Array.from((inputIds as any).data);
        } else {
            idsArray = Array.from(inputIds as any);
        }

        const seqLen = idsArray.length;
        const hiddenDim = steeringVector.length;

        // Scale the steering vector by strength
        const scaledVector = scale(steeringVector, strength);

        // Try to get input embeddings from the model
        let inputEmbeddings: Float32Array | null = null;

        try {
            // Method 1: Try to access the embedding layer directly
            const embedLayer = model.model?.embed_tokens ||
                model.transformer?.wte ||
                model.embeddings?.word_embeddings;

            if (embedLayer) {
                // Get embedding weights
                const weightData = embedLayer.weight?.data || embedLayer.weight?.ort_tensor?.cpuData;

                if (weightData) {
                    // Look up embeddings for each token
                    inputEmbeddings = new Float32Array(seqLen * hiddenDim);

                    for (let i = 0; i < seqLen; i++) {
                        const tokenId = Number(idsArray[i]);
                        const embStart = tokenId * hiddenDim;

                        for (let j = 0; j < hiddenDim; j++) {
                            // Copy original embedding
                            inputEmbeddings[i * hiddenDim + j] = Number(weightData[embStart + j]);
                            // Add steering vector to each position
                            inputEmbeddings[i * hiddenDim + j] += scaledVector[j];
                        }
                    }

                    console.log('✓ Created steered embeddings for', seqLen, 'tokens');
                }
            }
        } catch (e) {
            console.warn('Could not access embedding layer directly:', e);
        }

        // If we have steered embeddings, use them
        if (inputEmbeddings) {
            const embedsTensor = new Tensor(
                'float32',
                inputEmbeddings,
                [1, seqLen, hiddenDim]
            );

            // Create attention mask and position ids
            const attentionMaskData = new BigInt64Array(seqLen).fill(1n);
            const attentionMask = new Tensor('int64', attentionMaskData, [1, seqLen]);

            const positionIdsData = new BigInt64Array(seqLen);
            for (let i = 0; i < seqLen; i++) {
                positionIdsData[i] = BigInt(i);
            }
            const positionIds = new Tensor('int64', positionIdsData, [1, seqLen]);

            // Try to run generation with inputs_embeds
            // Note: transformers.js pipeline might not support inputs_embeds directly
            // We'll try calling the model directly and then decoding
            try {
                // First, get the model's output with steered embeddings
                const modelOutput = await model({
                    inputs_embeds: embedsTensor,
                    attention_mask: attentionMask,
                    position_ids: positionIds,
                    output_attentions: true,
                    output_hidden_states: true,
                });

                // Get the predicted next token from logits
                const logits = modelOutput.logits;
                let logitsData: Float32Array | number[];
                if (logits.data) {
                    logitsData = logits.data;
                } else if (logits.ort_tensor?.cpuData) {
                    logitsData = logits.ort_tensor.cpuData;
                } else {
                    throw new Error('Cannot access logits');
                }

                const vocabSize = logits.dims?.[logits.dims.length - 1] || 50257;
                const lastPosStart = (seqLen - 1) * vocabSize;
                const lastLogits = Array.from(logitsData.slice(lastPosStart, lastPosStart + vocabSize));

                // Apply temperature
                const temp = config.temperature || 1.0;
                const scaledLogits = lastLogits.map(l => Number(l) / temp);

                // Softmax
                const maxLogit = Math.max(...scaledLogits);
                const expLogits = scaledLogits.map(l => Math.exp(l - maxLogit));
                const sumExp = expLogits.reduce((a, b) => a + b, 0);
                const probs = expLogits.map(e => e / sumExp);

                // Sample or argmax based on temperature
                let nextTokenId: number;
                if (temp <= 0.01) {
                    // Greedy
                    nextTokenId = probs.indexOf(Math.max(...probs));
                } else {
                    // Top-k sampling
                    const k = config.topK || 50;
                    const indexed = probs.map((p, i) => ({ p, i }));
                    indexed.sort((a, b) => b.p - a.p);
                    const topK = indexed.slice(0, k);

                    // Re-normalize top-k
                    const topKSum = topK.reduce((s, x) => s + x.p, 0);
                    const topKProbs = topK.map(x => x.p / topKSum);

                    // Sample
                    const r = Math.random();
                    let cumsum = 0;
                    nextTokenId = topK[topK.length - 1].i;
                    for (let i = 0; i < topK.length; i++) {
                        cumsum += topKProbs[i];
                        if (r < cumsum) {
                            nextTokenId = topK[i].i;
                            break;
                        }
                    }
                }

                // Generate multiple tokens if requested
                let generatedIds: number[] = [...idsArray.map(Number), nextTokenId];
                let currentEmbeddings = inputEmbeddings;

                // Generate additional tokens
                for (let step = 1; step < config.maxNewTokens; step++) {
                    const currentSeqLen = generatedIds.length;

                    // Create new embeddings with the generated token
                    const newEmbeddings = new Float32Array(currentSeqLen * hiddenDim);

                    // Copy previous embeddings
                    for (let i = 0; i < currentSeqLen - 1; i++) {
                        for (let j = 0; j < hiddenDim; j++) {
                            newEmbeddings[i * hiddenDim + j] = currentEmbeddings[i * hiddenDim + j];
                        }
                    }

                    // Get embedding for new token and add steering
                    const embedLayer = model.model?.embed_tokens ||
                        model.transformer?.wte ||
                        model.embeddings?.word_embeddings;
                    const weightData = embedLayer?.weight?.data || embedLayer?.weight?.ort_tensor?.cpuData;

                    if (weightData) {
                        const lastTokenId = generatedIds[currentSeqLen - 1];
                        const embStart = lastTokenId * hiddenDim;
                        for (let j = 0; j < hiddenDim; j++) {
                            newEmbeddings[(currentSeqLen - 1) * hiddenDim + j] =
                                Number(weightData[embStart + j]) + scaledVector[j];
                        }
                    }

                    currentEmbeddings = newEmbeddings;

                    const newEmbedsTensor = new Tensor(
                        'float32',
                        newEmbeddings,
                        [1, currentSeqLen, hiddenDim]
                    );

                    const newAttnMask = new Tensor(
                        'int64',
                        new BigInt64Array(currentSeqLen).fill(1n),
                        [1, currentSeqLen]
                    );

                    const newPosIds = new BigInt64Array(currentSeqLen);
                    for (let i = 0; i < currentSeqLen; i++) {
                        newPosIds[i] = BigInt(i);
                    }
                    const newPosIdsTensor = new Tensor('int64', newPosIds, [1, currentSeqLen]);

                    const stepOutput = await model({
                        inputs_embeds: newEmbedsTensor,
                        attention_mask: newAttnMask,
                        position_ids: newPosIdsTensor,
                    });

                    // Get next token
                    let stepLogits: Float32Array | number[];
                    if (stepOutput.logits.data) {
                        stepLogits = stepOutput.logits.data;
                    } else if (stepOutput.logits.ort_tensor?.cpuData) {
                        stepLogits = stepOutput.logits.ort_tensor.cpuData;
                    } else {
                        break;
                    }

                    // Apply simple repetition penalty to logits before sampling
                    const repetitionPenalty = 1.2;
                    for (const id of generatedIds) {
                        const tokenIdx = Number(id);
                        if (tokenIdx < stepLogits.length) {
                            // If logit is positive, divide by penalty. If negative, multiply.
                            // Simplified: just decrease probability of seen tokens
                            if (stepLogits[tokenIdx] > 0) {
                                stepLogits[tokenIdx] /= repetitionPenalty;
                            } else {
                                stepLogits[tokenIdx] *= repetitionPenalty;
                            }
                        }
                    }

                    const stepVocabSize = stepOutput.logits.dims?.[stepOutput.logits.dims.length - 1] || 50257;
                    const stepLastStart = (currentSeqLen - 1) * stepVocabSize;
                    const stepLastLogits = Array.from(stepLogits.slice(stepLastStart, stepLastStart + stepVocabSize));

                    const stepScaledLogits = stepLastLogits.map(l => Number(l) / temp);
                    const stepMaxLogit = Math.max(...stepScaledLogits);
                    const stepExpLogits = stepScaledLogits.map(l => Math.exp(l - stepMaxLogit));
                    const stepSumExp = stepExpLogits.reduce((a, b) => a + b, 0);
                    const stepProbs = stepExpLogits.map(e => e / stepSumExp);

                    // Sample next token
                    let stepNextToken: number;
                    if (temp <= 0.01) {
                        stepNextToken = stepProbs.indexOf(Math.max(...stepProbs));
                    } else {
                        const k = config.topK || 50;
                        const indexed = stepProbs.map((p, i) => ({ p, i }));
                        indexed.sort((a, b) => b.p - a.p);
                        const topK = indexed.slice(0, k);
                        const topKSum = topK.reduce((s, x) => s + x.p, 0);
                        const topKProbs = topK.map(x => x.p / topKSum);

                        const r = Math.random();
                        let cumsum = 0;
                        stepNextToken = topK[topK.length - 1].i;
                        for (let i = 0; i < topK.length; i++) {
                            cumsum += topKProbs[i];
                            if (r < cumsum) {
                                stepNextToken = topK[i].i;
                                break;
                            }
                        }
                    }

                    generatedIds.push(stepNextToken);

                    // Check for EOS
                    if (stepNextToken === 50256) break; // GPT-2 EOS token
                }

                // Decode the full sequence
                const generatedText = tokenizer.decode(generatedIds, { skip_special_tokens: true });

                return [{
                    generated_text: generatedText,
                    _steered: true,
                    _modelOutput: modelOutput,
                }];
            } catch (e) {
                console.warn('Steered generation failed, falling back to normal:', e);
            }
        }

        // Fallback to normal generation if steering failed
        console.log('Falling back to normal generation');
        return await pipelineRef.current!(prompt, {
            max_new_tokens: config.maxNewTokens,
            temperature: config.temperature,
            top_k: config.topK,
            repetition_penalty: 1.2, // Encourages diversity
            return_full_text: true,
        });
    };

    // Unload model to free memory
    const unloadModel = useCallback(() => {
        pipelineRef.current = null;
        tokenizerRef.current = null;
        modelRef.current = null;
        embeddingMatrixRef.current = null;
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
        clearCache,
        getTokenizer: () => tokenizerRef.current,
        getHiddenState, // Already implemented in previous read
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
// Tries Q@K^T first (accurate), falls back to K@K^T (key similarity approximation)
function extractAttentionFromKV(modelOutput: any, seqLen: number): { attentions: number[][][][]; isQKT: boolean } | null {
    try {
        // Log available keys for debugging
        const keys = Object.keys(modelOutput);
        const kvKeys = keys.filter(k => k.includes('present') || k.includes('key') || k.includes('query') || k.includes('value'));
        console.log('Model output keys for KV extraction:', kvKeys);

        const attentions: number[][][][] = [];
        let layerIdx = 0;
        let usedQKT = false;

        // Try to determine the key format
        // Common formats: 'present.0.key', 'past_key_values.0.key', 'present_key_values.0.key'
        const hasPresentDot = keys.some(k => k.match(/^present\.\d+\.key$/));
        const hasPastDot = keys.some(k => k.match(/^past_key_values\.\d+\.key$/));

        console.log(`KV Key Format Detection: present.X.key=${hasPresentDot}, past_key_values.X.key=${hasPastDot}`);

        // Iterate through layers
        while (true) {
            let keyTensor: any = null;
            let queryTensor: any = null;

            if (hasPresentDot && modelOutput[`present.${layerIdx}.key`]) {
                keyTensor = modelOutput[`present.${layerIdx}.key`];
                queryTensor = modelOutput[`present.${layerIdx}.query`];
            } else if (hasPastDot && modelOutput[`past_key_values.${layerIdx}.key`]) {
                keyTensor = modelOutput[`past_key_values.${layerIdx}.key`];
                queryTensor = modelOutput[`past_key_values.${layerIdx}.query`];
            } else if (modelOutput[`present.${layerIdx}.key`]) {
                // Fallback to simple check
                keyTensor = modelOutput[`present.${layerIdx}.key`];
                queryTensor = modelOutput[`present.${layerIdx}.query`];
            } else if (modelOutput[`model.layers.${layerIdx}.self_attn.k_proj.weight`]) {
                // If we see weights but no KV cache, we might be looking at the wrong object or it's not outputting cache
                break;
            } else {
                // No more layers found
                break;
            }

            if (!keyTensor) break;

            // Get key tensor data and dimensions
            const keyData = keyTensor.data || keyTensor.ort_tensor?.cpuData;
            const keyDims = keyTensor.dims || keyTensor.shape;

            if (!keyData || !keyDims) {
                layerIdx++;
                continue;
            }

            // dims typically: [batch, num_heads, seq_len, head_dim]
            if (layerIdx === 0) console.log(`KV layer ${layerIdx} key dims:`, keyDims);

            if (keyDims.length < 3) { // Allow 3 dims if batch=1 is squeezed
                console.warn(`Unexpected dims length: ${keyDims.length}, skipping layer`);
                layerIdx++;
                continue;
            }

            // Handle different dimension layouts
            // [batch, heads, seq, dim] vs [batch, heads, dim, seq] vs [batch, seq, heads, dim]
            let numHeads: number, tensorSeqLen: number, headDim: number;

            if (keyDims.length === 4) {
                // Standard [batch, heads, seq, dim] for many ONNX (including Qwen/Llama usually)
                // BUT Qwen might be [batch, heads, seq, dim] or [batch, seq, heads, dim]?
                // Let's assume standard for now: [batch, heads, seq, dim]
                [, numHeads, tensorSeqLen, headDim] = keyDims;

                // Heuristic: if dim 2 is usually small (heads), dim 3 is variable (seq)?
                // Qwen 0.5B: 16 heads, 1024 hidden -> 64 head dim.
                // If we see [1, 16, 12, 64] -> heads=16, seq=12, dim=64.
                // If we see [1, 12, 16, 64] -> seq=12?

                // Check if likely transposed
                if (numHeads > 100 && tensorSeqLen < 100) {
                    // likely [batch, seq, heads, dim]
                    const tmp = numHeads;
                    numHeads = tensorSeqLen;
                    tensorSeqLen = tmp;
                }
            } else {
                // fallback
                layerIdx++;
                continue;
            }

            const actualSeqLen = Math.min(seqLen, tensorSeqLen);

            // Check if we have query tensor for real Q@K^T computation
            const queryData = queryTensor?.data || queryTensor?.ort_tensor?.cpuData;
            const hasQuery = queryData && queryData.length > 0;

            if (hasQuery && layerIdx === 0) {
                console.log(`✓ Query tensors available - using Q@K^T (real attention formula)`);
                usedQKT = true;
            }

            const layerAttention: number[][][] = [];

            for (let h = 0; h < numHeads; h++) {
                const headMatrix: number[][] = [];

                for (let i = 0; i < actualSeqLen; i++) {
                    const row: number[] = [];
                    for (let j = 0; j < actualSeqLen; j++) {
                        let dotProduct = 0;

                        if (hasQuery) {
                            // Compute Q[i] @ K[j]^T (real attention)
                            for (let d = 0; d < headDim; d++) {
                                const q_idx = h * tensorSeqLen * headDim + i * headDim + d;
                                const k_idx = h * tensorSeqLen * headDim + j * headDim + d;
                                dotProduct += Number(queryData[q_idx]) * Number(keyData[k_idx]);
                            }
                        } else {
                            // Fallback: Compute K[i] @ K[j]^T (key similarity approximation)
                            for (let d = 0; d < headDim; d++) {
                                const idx_i = h * tensorSeqLen * headDim + i * headDim + d;
                                const idx_j = h * tensorSeqLen * headDim + j * headDim + d;
                                dotProduct += Number(keyData[idx_i]) * Number(keyData[idx_j]);
                            }
                        }

                        // Apply causal mask and scale
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
            const method = usedQKT ? 'Q@K^T (real attention)' : 'K@K^T (key similarity approximation)';
            console.log(`Extracted ${attentions.length} layers from KV cache using ${method}`);
            return { attentions, isQKT: usedQKT };
        }

        return null;
    } catch (e) {
        console.error('KV extraction failed:', e);
        return null;
    }
}

// Get top predicted tokens from model logits
async function getTopPredictions(
    modelOutput: any,
    tokenizer: any,
    numLayers: number,
    explicitVocabSize?: number
): Promise<{ token: string; probability: number }[][]> {
    const predictions: { token: string; probability: number }[][] = [];

    try {
        // Get logits from model output - shape: [batch, seq_len, vocab_size]
        const logits = modelOutput?.logits;
        if (!logits) {
            console.warn('No logits in model output');
            return [];
        }

        // Get logits data
        let logitsData: Float32Array | number[];
        if (logits.data) {
            logitsData = logits.data;
        } else if (logits.ort_tensor?.cpuData) {
            logitsData = logits.ort_tensor.cpuData;
        } else {
            console.warn('Cannot access logits data');
            return [];
        }

        const dims = logits.dims || logits.shape || logits.ort_tensor?.dims || [];
        const totalLen = logitsData.length;

        let vocabSize = 0;
        let seqLen = 0;

        // 1. Trust Tensor Dimensions (If Valid)
        if (dims.length >= 1) {
            const lastDim = dims[dims.length - 1];
            if (lastDim > 30000) {
                vocabSize = lastDim;
                seqLen = Math.floor(totalLen / vocabSize);
                console.log(`[getTopPredictions] Found dims: ${dims} -> vocab=${vocabSize}, seq=${seqLen}`);
            }
        }

        // 2. Adaptive Snapping (If dims failed or missing)
        if (vocabSize === 0) {
            const commonVocabs = [151936, 151643, 151646, 128256, 32000, 32001, 32064, 50257, 50272, 50277];

            // FORCE Explicit Size if available (e.g. Qwen 151936)
            // Even if totalLen isn't perfectly divisible, we trust the known model architecture
            if (explicitVocabSize && explicitVocabSize > 0) {
                vocabSize = explicitVocabSize;
                seqLen = Math.floor(totalLen / vocabSize);
                console.log(`[getTopPredictions] Forcing explicit vocab size: ${vocabSize} (seq=${seqLen})`);
            } else {
                // Try to find a matching vocab
                for (const v of commonVocabs) {
                    if (totalLen % v === 0) {
                        vocabSize = v;
                        seqLen = totalLen / v;
                        console.log(`[getTopPredictions] Auto-detected vocab: ${vocabSize}`);
                        break;
                    }
                }
            }
        }

        // 3. Last Resort: Single Token Output?
        if (vocabSize === 0) {
            if (totalLen > 30000) {
                vocabSize = totalLen;
                seqLen = 1;
                console.log(`[getTopPredictions] Fallback: Assuming single token output (vocab=${vocabSize})`);
            } else {
                vocabSize = 50257; // Legacy GPT-2
                seqLen = Math.floor(totalLen / vocabSize);
                console.log(`[getTopPredictions] Fallback: Legacy GPT-2 default`);
            }
        }

        // Safety clamp
        if (seqLen === 0) seqLen = 1;

        console.log(`[getTopPredictions] Final: seq=${seqLen}, vocab=${vocabSize}, total=${totalLen}`);

        // Get logits for the LAST position
        const lastPosStart = (seqLen - 1) * vocabSize;

        // RELAXED Bounds Check:
        // If data is truncated (common in quantized models), we still want to read what's there.
        if (lastPosStart < 0) {
            console.error(`[getTopPredictions] Negative start index: ${lastPosStart}`);
            return [];
        }

        if (lastPosStart + vocabSize > totalLen) {
            console.warn(`[getTopPredictions] Truncated data: Expected ${vocabSize}, available ${totalLen - lastPosStart}. Slicing safe range.`);
        }

        // slice() handles out-of-bounds end automatically by clamping to length
        const lastLogits = Array.from(logitsData.slice(lastPosStart, lastPosStart + vocabSize));

        // Compute softmax probabilities
        const maxLogit = Math.max(...lastLogits);
        const expLogits = lastLogits.map(l => Math.exp(Number(l) - maxLogit));
        const sumExp = expLogits.reduce((a, b) => a + b, 0);
        const probs = expLogits.map(e => e / sumExp);

        // Get top 5 token indices
        const indexed = probs.map((p, i) => ({ prob: p, idx: i }));
        indexed.sort((a, b) => b.prob - a.prob);
        const top5 = indexed.slice(0, 5);

        // Build reverse vocab map for fast lookup
        let reverseVocab: Map<number, string> | null = null;
        if (tokenizer.model?.vocab) {
            reverseVocab = new Map();
            for (const [token, id] of tokenizer.model.vocab) {
                reverseVocab.set(Number(id), token);
            }
        }

        // Decode token IDs to strings
        const topPredictions: { token: string; probability: number }[] = [];
        for (const { prob, idx } of top5) {
            let tokenStr = `[${idx}]`;
            try {
                // Fast vocab lookup
                if (reverseVocab?.has(idx)) {
                    tokenStr = reverseVocab.get(idx)!;
                    // Clean up GPT-2 space markers
                    tokenStr = tokenStr.replace(/^Ġ/, ' ').replace(/^▁/, ' ');
                } else {
                    // Fallback to decode
                    const decoded = tokenizer.decode([idx], { skip_special_tokens: false });
                    if (decoded && decoded.length > 0) {
                        tokenStr = decoded;
                    }
                }
            } catch (e) {
                console.warn(`Failed to decode token ${idx}:`, e);
            }

            topPredictions.push({
                token: tokenStr,
                probability: prob
            });
        }

        // For now, show same predictions for all "layers" since we only have final logits
        // True logit lens would require intermediate layer hidden states
        for (let layer = 0; layer < numLayers; layer++) {
            predictions.push(topPredictions);
        }

        console.log('Top predictions:', topPredictions);

    } catch (e) {
        console.warn('Could not get predictions:', e);
        // Debug probe
        predictions.push([{ token: `Err: ${e}`, probability: 0 }]);
    }

    if (predictions.length === 0) {
        // Debug probe for empty return
        predictions.push([{ token: "No Data", probability: 0 }]);
    }

    return predictions;
}

// Extract hidden states from model output
async function extractHiddenStates(hiddenStatesOutput: any[]): Promise<number[][][]> {
    const hiddenStates: number[][][] = [];

    for (const layerHidden of hiddenStatesOutput) {
        let data: Float32Array | number[];
        if (layerHidden.data) {
            data = layerHidden.data;
        } else if (layerHidden.ort_tensor?.cpuData) {
            data = layerHidden.ort_tensor.cpuData;
        } else {
            data = await layerHidden.tolist();
        }

        const dims = layerHidden.dims || layerHidden.shape || [1, 0, 0];
        const [_batch, seqLen, hiddenDim] = dims;

        if (!seqLen || !hiddenDim) continue;

        const layerData: number[][] = [];
        for (let i = 0; i < seqLen; i++) {
            const row: number[] = [];
            for (let j = 0; j < hiddenDim; j++) {
                const idx = i * hiddenDim + j;
                row.push(Number(data[idx]) || 0);
            }
            layerData.push(row);
        }
        hiddenStates.push(layerData);
    }

    return hiddenStates;
}

// Extract raw logits for the final position
function extractRawLogits(logits: any, explicitVocabSize?: number): number[] {
    let logitsData: Float32Array | number[];
    if (logits.data) {
        logitsData = logits.data;
    } else if (logits.ort_tensor?.cpuData) {
        logitsData = logits.ort_tensor.cpuData;
    } else {
        return [];
    }

    const dims = logits.dims || logits.shape || [];
    const vocabSize = explicitVocabSize || dims[dims.length - 1] || 50257;
    let seqLen = dims.length > 2 ? dims[1] : 1;

    if (dims.length === 0 && logitsData.length > 0) {
        seqLen = Math.floor(logitsData.length / vocabSize);
    }

    // Get logits for the last position
    const lastPosStart = (seqLen - 1) * vocabSize;
    if (lastPosStart < 0 || lastPosStart + vocabSize > logitsData.length) return [];

    return Array.from(logitsData.slice(lastPosStart, lastPosStart + vocabSize)).map(Number);
}
