import type { CircuitInfo } from '../types';

// Detect interpretable circuits in attention patterns
export function detectCircuits(
    attentions: number[][][][],
    tokens: string[]
): CircuitInfo[] {
    const circuits: CircuitInfo[] = [];

    if (!attentions.length || !tokens.length) return circuits;

    const numLayers = attentions.length;
    const numHeads = attentions[0]?.length || 0;

    for (let layer = 0; layer < numLayers; layer++) {
        for (let head = 0; head < numHeads; head++) {
            const attnMatrix = attentions[layer]?.[head];
            if (!attnMatrix) continue;

            // Check for previous token head
            const prevTokenScore = detectPreviousTokenHead(attnMatrix);
            if (prevTokenScore > 0.4) {
                circuits.push({
                    type: 'previous_token',
                    layer,
                    head,
                    score: prevTokenScore,
                    explanation: `Head L${layer}H${head} primarily attends to the previous token. This helps the model understand word order and local context.`,
                });
            }

            // Check for induction head pattern
            const inductionScore = detectInductionHead(attnMatrix, tokens);
            if (inductionScore > 0.3) {
                circuits.push({
                    type: 'induction',
                    layer,
                    head,
                    score: inductionScore,
                    explanation: `Head L${layer}H${head} shows induction behavior - it copies tokens that followed similar previous tokens. This enables in-context learning.`,
                });
            }

            // Check for duplicate token attention
            const dupScore = detectDuplicateTokenHead(attnMatrix, tokens);
            if (dupScore > 0.35) {
                circuits.push({
                    type: 'duplicate_token',
                    layer,
                    head,
                    score: dupScore,
                    explanation: `Head L${layer}H${head} attends to duplicate tokens in the sequence. This helps track repeated words or patterns.`,
                });
            }
        }
    }

    // Sort by score descending
    return circuits.sort((a, b) => b.score - a.score);
}

// Detect heads that primarily attend to previous token
function detectPreviousTokenHead(attnMatrix: number[][]): number {
    const seqLen = attnMatrix.length;
    if (seqLen < 2) return 0;

    let diagSum = 0;
    let count = 0;

    // Sum attention on the -1 diagonal (previous token)
    for (let i = 1; i < seqLen; i++) {
        diagSum += attnMatrix[i][i - 1];
        count++;
    }

    return count > 0 ? diagSum / count : 0;
}

// Detect induction heads via pattern matching
function detectInductionHead(attnMatrix: number[][], tokens: string[]): number {
    const seqLen = tokens.length;
    if (seqLen < 4) return 0;

    let inductionScore = 0;
    let pairs = 0;

    // Look for [A][B]...[A] pattern where position after second A attends to B
    for (let i = 2; i < seqLen; i++) {
        for (let j = 0; j < i - 1; j++) {
            // Check if tokens match
            if (tokens[i].toLowerCase() === tokens[j].toLowerCase()) {
                // Position after current should attend to position after match
                if (i + 1 < seqLen && j + 1 < i) {
                    inductionScore += attnMatrix[i][j + 1];
                    pairs++;
                }
            }
        }
    }

    return pairs > 0 ? inductionScore / pairs : 0;
}

// Detect heads that attend to duplicate tokens
function detectDuplicateTokenHead(attnMatrix: number[][], tokens: string[]): number {
    const seqLen = tokens.length;
    if (seqLen < 2) return 0;

    let dupScore = 0;
    let pairs = 0;

    for (let i = 1; i < seqLen; i++) {
        for (let j = 0; j < i; j++) {
            if (tokens[i].toLowerCase() === tokens[j].toLowerCase()) {
                dupScore += attnMatrix[i][j];
                pairs++;
            }
        }
    }

    return pairs > 0 ? dupScore / pairs : 0;
}

// Get human-readable circuit type name
export function getCircuitTypeName(type: CircuitInfo['type']): string {
    const names: Record<CircuitInfo['type'], string> = {
        induction: 'Induction Head',
        previous_token: 'Previous Token Head',
        duplicate_token: 'Duplicate Token Head',
        unknown: 'Unknown Pattern',
    };
    return names[type];
}
