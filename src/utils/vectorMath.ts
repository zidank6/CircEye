/**
 * Math utilities for Float32Array vector operations
 * used for model steering and activation engineering.
 */

// Create a zero-filled vector
export function zeros(length: number): Float32Array {
    return new Float32Array(length);
}

// Add two vectors: a + b
export function add(a: Float32Array, b: Float32Array): Float32Array {
    if (a.length !== b.length) throw new Error('Vector length mismatch');
    const result = new Float32Array(a.length);
    for (let i = 0; i < a.length; i++) {
        result[i] = a[i] + b[i];
    }
    return result;
}

// Subtract two vectors: a - b
export function subtract(a: Float32Array, b: Float32Array): Float32Array {
    if (a.length !== b.length) throw new Error('Vector length mismatch');
    const result = new Float32Array(a.length);
    for (let i = 0; i < a.length; i++) {
        result[i] = a[i] - b[i];
    }
    return result;
}

// Scale a vector by a scalar: v * s
export function scale(v: Float32Array, s: number): Float32Array {
    const result = new Float32Array(v.length);
    for (let i = 0; i < v.length; i++) {
        result[i] = v[i] * s;
    }
    return result;
}

// Compute the mean vector from an array of vectors
export function mean(vectors: Float32Array[]): Float32Array {
    if (vectors.length === 0) throw new Error('Cannot compute mean of empty list');
    const dim = vectors[0].length;
    const sum = new Float32Array(dim);

    for (const v of vectors) {
        if (v.length !== dim) throw new Error('Vector length mismatch in mean computation');
        for (let i = 0; i < dim; i++) {
            sum[i] += v[i];
        }
    }

    return scale(sum, 1 / vectors.length);
}

// Compute steering vector: Mean(Pos) - Mean(Neg)
export function computeSteeringVector(
    positives: Float32Array[],
    negatives: Float32Array[]
): Float32Array {
    const posMean = mean(positives);
    const negMean = mean(negatives);
    return subtract(posMean, negMean);
}

// Serialize vector to Base64 string for storage
export function serializeVector(v: Float32Array): string {
    // Convert Float32Array to Uint8Array buffer
    const buffer = new Uint8Array(v.buffer);
    // Convert buffer to binary string
    let binary = '';
    for (let i = 0; i < buffer.byteLength; i++) {
        binary += String.fromCharCode(buffer[i]);
    }
    // Base64 encode
    return btoa(binary);
}

// Deserialize Base64 string to Float32Array
export function deserializeVector(base64: string): Float32Array {
    const binary = atob(base64);
    const buffer = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
        buffer[i] = binary.charCodeAt(i);
    }
    return new Float32Array(buffer.buffer);
}

export interface SteeringVectorData {
    id: string;
    name: string;
    description?: string;
    vectorBase64: string;
    dimension: number;
    createdAt: number;
}

export function createSteeringVectorData(
    name: string,
    vector: Float32Array,
    description?: string
): SteeringVectorData {
    return {
        id: crypto.randomUUID(),
        name,
        description,
        vectorBase64: serializeVector(vector),
        dimension: vector.length,
        createdAt: Date.now(),
    };
}

// Load steering vector from saved data
export function loadSteeringVector(data: SteeringVectorData): Float32Array {
    return deserializeVector(data.vectorBase64);
}
