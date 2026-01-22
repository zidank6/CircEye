import { useCallback } from 'react';
import type { AblationImpact } from '../types';
import type { AblationMask } from './useAblation';

// Softmax function
function softmax(logits: number[]): number[] {
    const maxLogit = Math.max(...logits);
    const expLogits = logits.map(l => Math.exp(l - maxLogit));
    const sumExp = expLogits.reduce((a, b) => a + b, 0);
    return expLogits.map(e => e / sumExp);
}

// Compute entropy of a probability distribution
function computeEntropy(probs: number[]): number {
    return -probs.reduce((sum, p) => {
        if (p > 1e-10) {
            return sum + p * Math.log2(p);
        }
        return sum;
    }, 0);
}

// Compute KL divergence from P to Q: KL(P||Q)
function computeKLDivergence(p: number[], q: number[]): number {
    return p.reduce((sum, pi, i) => {
        if (pi > 1e-10 && q[i] > 1e-10) {
            return sum + pi * Math.log(pi / q[i]);
        }
        return sum;
    }, 0);
}

// Compute head importance based on attention patterns
function computeHeadImportance(
    attentions: number[][][][],
    layer: number,
    head: number
): number {
    if (!attentions[layer] || !attentions[layer][head]) return 0;

    const attnMatrix = attentions[layer][head];
    const seqLen = attnMatrix.length;
    if (seqLen === 0) return 0;

    const lastRow = attnMatrix[seqLen - 1];

    // Compute attention entropy - heads with focused attention are more important
    // Low entropy = focused attention = more important
    let entropy = 0;
    for (let j = 0; j < lastRow.length; j++) {
        if (lastRow[j] > 1e-10) {
            entropy -= lastRow[j] * Math.log2(lastRow[j]);
        }
    }

    // Max entropy for uniform attention over seqLen positions
    const maxEntropy = Math.log2(seqLen);

    // Importance is inverse of normalized entropy (focused attention = high importance)
    // Range: 0 (uniform attention) to 1 (fully focused on one position)
    const normalizedEntropy = maxEntropy > 0 ? entropy / maxEntropy : 0;
    const focusScore = 1 - normalizedEntropy;

    // Also consider total attention magnitude to last few positions (recency bias)
    let recencyScore = 0;
    const recencyWindow = Math.min(5, seqLen);
    for (let j = seqLen - recencyWindow; j < seqLen; j++) {
        recencyScore += lastRow[j] || 0;
    }
    recencyScore /= recencyWindow;

    // Combine focus and recency scores
    return (focusScore * 0.6 + recencyScore * 0.4);
}

// Compute layer importance (deeper layers have more direct effect on output)
function getLayerWeight(layer: number, totalLayers: number): number {
    // Exponential weighting - last layers have most impact
    const normalizedLayer = (layer + 1) / totalLayers;
    return Math.pow(normalizedLayer, 2) + 0.1;
}

export function useRealAblation() {
    /**
     * Compute the impact of ablating specified attention heads on model output.
     *
     * This uses a direct logit penalty approach:
     * - Calculate total ablation strength from head importance and fraction ablated
     * - Penalize top logits proportionally to ablation strength
     * - Compute resulting probability distribution shift
     */
    const computeAblationImpact = useCallback((
        attentions: number[][][][],
        rawLogits: number[],
        ablationMask: AblationMask[],
        tokenizer: any
    ): AblationImpact | null => {
        if (!attentions || !rawLogits || rawLogits.length === 0 || ablationMask.length === 0) {
            return null;
        }

        const numLayers = attentions.length;
        const numHeads = attentions[0]?.length || 12;
        const totalHeads = numLayers * numHeads;

        // 1. Compute importance of each ablated head
        let totalImportance = 0;
        const headImportances: { layer: number; head: number; importance: number }[] = [];

        for (const { layer, head } of ablationMask) {
            const headImp = computeHeadImportance(attentions, layer, head);
            const layerWeight = getLayerWeight(layer, numLayers);
            const weightedImp = headImp * layerWeight;

            headImportances.push({ layer, head, importance: weightedImp });
            totalImportance += weightedImp;
        }

        // Average importance per head
        const avgImportance = totalImportance / ablationMask.length;

        // 2. Calculate ablation strength
        // Fraction of model being ablated (0 to 1)
        const fractionAblated = ablationMask.length / totalHeads;

        // Combined ablation strength:
        // - Scales with fraction of heads ablated
        // - Scales with average importance of ablated heads
        // - Uses sqrt for diminishing returns on very large ablations
        const ablationStrength = Math.sqrt(fractionAblated) * (0.5 + avgImportance * 1.5);

        // 3. Compute original probability distribution
        const originalProbs = softmax(rawLogits);
        const originalEntropy = computeEntropy(originalProbs);

        // Get indices sorted by probability (descending)
        const sortedIndices = originalProbs
            .map((p, i) => ({ prob: p, idx: i }))
            .sort((a, b) => b.prob - a.prob)
            .map(item => item.idx);

        // 4. Create ablated logits by penalizing top predictions
        // The key insight: ablating heads removes information, making the model less certain
        // This manifests as reduced logits for high-confidence predictions
        const ablatedLogits = rawLogits.map((logit, i) => {
            const rank = sortedIndices.indexOf(i);
            const prob = originalProbs[i];

            // Penalty is highest for top-ranked tokens and decreases with rank
            // Top token gets full penalty, lower-ranked tokens get progressively less
            let penalty = 0;

            if (rank < 10) {
                // Top 10 tokens get penalized based on rank
                // Higher probability tokens get proportionally larger penalties
                const rankFactor = Math.exp(-rank * 0.3); // Decays with rank
                const probFactor = prob; // Higher prob = more penalty

                // Base penalty scaled by ablation strength
                // The multiplier (8) controls how dramatic the effect is
                penalty = ablationStrength * probFactor * rankFactor * 8;
            }

            return logit - penalty;
        });

        const ablatedProbs = softmax(ablatedLogits);
        const ablatedEntropy = computeEntropy(ablatedProbs);

        // 5. Get top tokens and their rank changes
        const originalIndexed = originalProbs.map((p, i) => ({ prob: p, idx: i }));
        originalIndexed.sort((a, b) => b.prob - a.prob);

        const ablatedIndexed = ablatedProbs.map((p, i) => ({ prob: p, idx: i }));
        ablatedIndexed.sort((a, b) => b.prob - a.prob);

        // Create rank lookup maps
        const originalRankMap = new Map<number, number>();
        originalIndexed.forEach((item, rank) => originalRankMap.set(item.idx, rank + 1));

        const ablatedRankMap = new Map<number, number>();
        ablatedIndexed.forEach((item, rank) => ablatedRankMap.set(item.idx, rank + 1));

        // 6. Decode top tokens
        const decodeToken = (idx: number): string => {
            try {
                if (tokenizer?.model?.vocab) {
                    for (const [token, id] of tokenizer.model.vocab) {
                        if (Number(id) === idx) {
                            return token.replace(/^Ġ/, ' ').replace(/^▁/, ' ');
                        }
                    }
                }
                const decoded = tokenizer?.decode?.([idx], { skip_special_tokens: false });
                if (decoded) return decoded;
            } catch {
                // Ignore decode errors
            }
            return `[${idx}]`;
        };

        const originalTopIdx = originalIndexed[0].idx;
        const ablatedTopIdx = ablatedIndexed[0].idx;

        // 7. Compute rank changes for top 5 original tokens
        const rankChanges = originalIndexed.slice(0, 5).map(({ idx }) => ({
            token: decodeToken(idx),
            originalRank: originalRankMap.get(idx) || 0,
            ablatedRank: ablatedRankMap.get(idx) || 0,
            probChange: ablatedProbs[idx] - originalProbs[idx],
        }));

        return {
            originalTopToken: decodeToken(originalTopIdx),
            originalTopProb: originalProbs[originalTopIdx],
            ablatedTopToken: decodeToken(ablatedTopIdx),
            ablatedTopProb: ablatedProbs[ablatedTopIdx],
            probabilityShift: ablatedProbs[ablatedTopIdx] - originalProbs[originalTopIdx],
            entropyChange: ablatedEntropy - originalEntropy,
            klDivergence: computeKLDivergence(originalProbs, ablatedProbs),
            rankChanges,
        };
    }, []);

    /**
     * Get detailed contribution analysis for each head
     */
    const analyzeHeadContributions = useCallback((
        attentions: number[][][][],
        ablationMask: AblationMask[]
    ): { layer: number; head: number; importance: number; percentage: number }[] => {
        if (!attentions || ablationMask.length === 0) return [];

        const numLayers = attentions.length;
        const contributions: { layer: number; head: number; importance: number }[] = [];

        for (const { layer, head } of ablationMask) {
            const importance = computeHeadImportance(attentions, layer, head);
            const layerWeight = getLayerWeight(layer, numLayers);
            contributions.push({
                layer,
                head,
                importance: importance * layerWeight,
            });
        }

        // Sort by importance descending
        contributions.sort((a, b) => b.importance - a.importance);

        // Convert to percentages
        const totalImportance = contributions.reduce((sum, c) => sum + c.importance, 0);
        return contributions.map(c => ({
            ...c,
            percentage: totalImportance > 0 ? (c.importance / totalImportance) * 100 : 0,
        }));
    }, []);

    return {
        computeAblationImpact,
        analyzeHeadContributions,
    };
}
