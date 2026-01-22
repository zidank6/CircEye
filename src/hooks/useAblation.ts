import { useState, useCallback } from 'react';
import type { CircuitInfo } from '../types';

export interface AblationMask {
    layer: number;
    head: number;
}

export interface AblationResult {
    originalAttentions: number[][][][] | null;
    ablatedAttentions: number[][][][] | null;
    ablatedHeads: AblationMask[];
    impactScore: number; // How much the output changed (0-1)
}

// Apply ablation by zeroing out specified heads
export function applyAblationToAttention(
    attentions: number[][][][],
    ablationMask: AblationMask[]
): number[][][][] {
    if (!attentions || !attentions.length) return attentions;

    // Deep copy to avoid mutating original
    const ablated = attentions.map(layer =>
        layer.map(head =>
            head.map(row => [...row])
        )
    );

    // Zero out ablated heads
    for (const { layer, head } of ablationMask) {
        if (ablated[layer] && ablated[layer][head]) {
            // Set to uniform attention instead of zero to maintain valid distribution
            ablated[layer][head] = ablated[layer][head].map((row, i) =>
                row.map((_, j) => j <= i ? 1 / (i + 1) : 0)
            );
        }
    }

    return ablated;
}

// Calculate how different two attention patterns are
export function calculateImpactScore(
    original: number[][][][],
    ablated: number[][][][]
): number {
    if (!original || !ablated) return 0;

    let totalDiff = 0;
    let count = 0;

    for (let l = 0; l < original.length && l < ablated.length; l++) {
        for (let h = 0; h < original[l].length && h < ablated[l].length; h++) {
            const origMatrix = original[l][h];
            const ablMatrix = ablated[l][h];
            if (!origMatrix || !ablMatrix) continue;

            for (let i = 0; i < origMatrix.length && i < ablMatrix.length; i++) {
                for (let j = 0; j < origMatrix[i].length && j < ablMatrix[i].length; j++) {
                    totalDiff += Math.abs((origMatrix[i][j] || 0) - (ablMatrix[i][j] || 0));
                    count++;
                }
            }
        }
    }

    return count > 0 ? Math.min(1, totalDiff / count * 2) : 0;
}

export function useAblation() {
    const [ablationResult, setAblationResult] = useState<AblationResult | null>(null);
    const [isComparing, setIsComparing] = useState(false);

    const runAblation = useCallback((
        originalAttentions: number[][][][],
        headMask: AblationMask[]
    ) => {
        if (!originalAttentions || headMask.length === 0) {
            setAblationResult(null);
            return null;
        }

        const ablated = applyAblationToAttention(originalAttentions, headMask);
        const impact = calculateImpactScore(originalAttentions, ablated);

        const result: AblationResult = {
            originalAttentions,
            ablatedAttentions: ablated,
            ablatedHeads: headMask,
            impactScore: impact
        };

        setAblationResult(result);
        setIsComparing(true);
        return result;
    }, []);

    const clearAblation = useCallback(() => {
        setAblationResult(null);
        setIsComparing(false);
    }, []);

    // Auto-ablate detected circuits to see their importance
    const ablateCircuits = useCallback((
        originalAttentions: number[][][][],
        circuits: CircuitInfo[]
    ) => {
        const headMask = circuits.map(c => ({
            layer: c.layer,
            head: c.head
        }));
        return runAblation(originalAttentions, headMask);
    }, [runAblation]);

    return {
        ablationResult,
        isComparing,
        runAblation,
        clearAblation,
        ablateCircuits
    };
}
