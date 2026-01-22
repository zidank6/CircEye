import type { CircuitInfo } from '../types';

export interface DetectedCircuit extends CircuitInfo {
    confidence: 'high' | 'medium' | 'low';
    evidence: string;
}

// Detect interpretable circuits in attention patterns
export function detectCircuits(
    attentions: number[][][][],
    tokens: string[]
): DetectedCircuit[] {
    const circuits: DetectedCircuit[] = [];

    if (!attentions || !attentions.length || !tokens || !tokens.length) return circuits;

    const numLayers = attentions.length;
    const numHeads = attentions[0]?.length || 0;

    if (numHeads === 0) return circuits;

    console.log(`Circuit detection: ${numLayers} layers, ${numHeads} heads, ${tokens.length} tokens`);

    for (let layer = 0; layer < numLayers; layer++) {
        for (let head = 0; head < numHeads; head++) {
            const attnMatrix = attentions[layer]?.[head];
            if (!attnMatrix) continue;

            // Check for previous token head
            const prevTokenResult = detectPreviousTokenHead(attnMatrix);
            if (prevTokenResult.score > 0.3) {
                circuits.push({
                    type: 'previous_token',
                    layer,
                    head,
                    score: prevTokenResult.score,
                    confidence: getConfidence(prevTokenResult.score, 0.3, 0.5, 0.7),
                    evidence: prevTokenResult.evidence,
                    explanation: `Head L${layer}H${head} primarily attends to the previous token (avg ${(prevTokenResult.score * 100).toFixed(0)}% attention). This helps the model understand word order and local context.`,
                });
            }

            // Check for induction head pattern
            const inductionResult = detectInductionHead(attnMatrix, tokens);
            if (inductionResult.score > 0.2) {
                circuits.push({
                    type: 'induction',
                    layer,
                    head,
                    score: inductionResult.score,
                    confidence: getConfidence(inductionResult.score, 0.2, 0.35, 0.5),
                    evidence: inductionResult.evidence,
                    explanation: `Head L${layer}H${head} shows induction behavior - it copies tokens that followed similar previous tokens. ${inductionResult.evidence}`,
                });
            }

            // Check for duplicate token attention
            const dupResult = detectDuplicateTokenHead(attnMatrix, tokens);
            if (dupResult.score > 0.25) {
                circuits.push({
                    type: 'duplicate_token',
                    layer,
                    head,
                    score: dupResult.score,
                    confidence: getConfidence(dupResult.score, 0.25, 0.4, 0.55),
                    evidence: dupResult.evidence,
                    explanation: `Head L${layer}H${head} attends to duplicate tokens in the sequence. ${dupResult.evidence}`,
                });
            }
        }
    }

    // Sort by score descending
    return circuits.sort((a, b) => b.score - a.score);
}

function getConfidence(score: number, _low: number, med: number, high: number): 'high' | 'medium' | 'low' {
    if (score >= high) return 'high';
    if (score >= med) return 'medium';
    return 'low';
}

interface DetectionResult {
    score: number;
    evidence: string;
}

// Detect heads that primarily attend to previous token
function detectPreviousTokenHead(attnMatrix: number[][]): DetectionResult {
    const seqLen = attnMatrix.length;
    if (seqLen < 2) return { score: 0, evidence: '' };

    let diagSum = 0;
    let count = 0;
    let maxAttn = 0;
    let maxPos = -1;

    // Sum attention on the -1 diagonal (previous token)
    for (let i = 1; i < seqLen; i++) {
        const row = attnMatrix[i];
        if (!row || row[i - 1] === undefined) continue;
        const val = row[i - 1];
        diagSum += val;
        count++;
        if (val > maxAttn) {
            maxAttn = val;
            maxPos = i;
        }
    }

    const avgScore = count > 0 ? diagSum / count : 0;
    const evidence = maxPos >= 0
        ? `Position ${maxPos} attends ${(maxAttn * 100).toFixed(0)}% to position ${maxPos - 1}`
        : '';

    return { score: avgScore, evidence };
}

// Detect induction heads via pattern matching
function detectInductionHead(attnMatrix: number[][], tokens: string[]): DetectionResult {
    const seqLen = Math.min(tokens.length, attnMatrix.length);
    if (seqLen < 4) return { score: 0, evidence: '' };

    let inductionScore = 0;
    let pairs = 0;
    let bestExample = '';
    let bestAttn = 0;

    // Look for [A][B]...[A] pattern where position after second A attends to B
    for (let i = 2; i < seqLen; i++) {
        const row = attnMatrix[i];
        if (!row) continue;

        for (let j = 0; j < i - 1; j++) {
            // Check if tokens match
            const tokenI = tokens[i]?.toLowerCase()?.trim();
            const tokenJ = tokens[j]?.toLowerCase()?.trim();
            if (tokenI && tokenJ && tokenI === tokenJ) {
                // Current position should attend to position after match
                if (j + 1 < i && row[j + 1] !== undefined) {
                    const attnVal = row[j + 1];
                    inductionScore += attnVal;
                    pairs++;
                    if (attnVal > bestAttn) {
                        bestAttn = attnVal;
                        bestExample = `"${tokens[i]}" at pos ${i} attends ${(attnVal * 100).toFixed(0)}% to pos ${j + 1} (after prev "${tokens[j]}")`;
                    }
                }
            }
        }
    }

    const avgScore = pairs > 0 ? inductionScore / pairs : 0;
    return {
        score: avgScore,
        evidence: bestExample || (pairs > 0 ? `Found ${pairs} potential induction patterns` : '')
    };
}

// Detect heads that attend to duplicate tokens
function detectDuplicateTokenHead(attnMatrix: number[][], tokens: string[]): DetectionResult {
    const seqLen = Math.min(tokens.length, attnMatrix.length);
    if (seqLen < 2) return { score: 0, evidence: '' };

    let dupScore = 0;
    let pairs = 0;
    let bestExample = '';
    let bestAttn = 0;

    for (let i = 1; i < seqLen; i++) {
        const row = attnMatrix[i];
        if (!row) continue;

        for (let j = 0; j < i; j++) {
            const tokenI = tokens[i]?.toLowerCase()?.trim();
            const tokenJ = tokens[j]?.toLowerCase()?.trim();
            if (tokenI && tokenJ && tokenI === tokenJ && row[j] !== undefined) {
                const attnVal = row[j];
                dupScore += attnVal;
                pairs++;
                if (attnVal > bestAttn) {
                    bestAttn = attnVal;
                    bestExample = `"${tokens[i]}" at pos ${i} attends ${(attnVal * 100).toFixed(0)}% to same token at pos ${j}`;
                }
            }
        }
    }

    const avgScore = pairs > 0 ? dupScore / pairs : 0;
    return {
        score: avgScore,
        evidence: bestExample || (pairs > 0 ? `Found ${pairs} duplicate token patterns` : '')
    };
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

// Get circuit type color for visualization
export function getCircuitColor(type: CircuitInfo['type']): string {
    const colors: Record<CircuitInfo['type'], string> = {
        induction: '#4ecdc4',
        previous_token: '#45b7d1',
        duplicate_token: '#f7dc6f',
        unknown: '#95a5a6',
    };
    return colors[type];
}

// Get confidence badge style
export function getConfidenceBadge(confidence: 'high' | 'medium' | 'low'): { color: string; label: string } {
    const badges = {
        high: { color: '#27ae60', label: 'High confidence' },
        medium: { color: '#f39c12', label: 'Medium confidence' },
        low: { color: '#e74c3c', label: 'Low confidence' }
    };
    return badges[confidence];
}
