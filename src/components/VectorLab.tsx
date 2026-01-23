import { useState } from 'react';
import {
    createSteeringVectorData,
    computeSteeringVector,
    type SteeringVectorData
} from '../utils/vectorMath';
import { message } from '@tauri-apps/plugin-dialog';

interface VectorLabProps {
    onCompute: (input: string) => Promise<Float32Array | null>;
    onVectorCreated: (vectorData: SteeringVectorData) => void;
    savedVectors: SteeringVectorData[];
    onLoadVector: (vector: SteeringVectorData) => void;
    onDeleteVector: (id: string) => void;
}

export function VectorLab({
    onCompute,
    onVectorCreated,
    savedVectors,
    onLoadVector,
    onDeleteVector
}: VectorLabProps) {
    const [posExamples, setPosExamples] = useState('Love\nPeace\nKindness\nJoy');
    const [negExamples, setNegExamples] = useState('Hate\nWar\nCruelty\nMisery');
    const [isComputing, setIsComputing] = useState(false);
    const [progress, setProgress] = useState(0);
    const [computedVector, setComputedVector] = useState<Float32Array | null>(null);
    const [vectorName, setVectorName] = useState('');

    const handleCompute = async () => {
        setIsComputing(true);
        setProgress(0);
        setComputedVector(null);

        try {
            const posLines = posExamples.split('\n').filter(l => l.trim());
            const negLines = negExamples.split('\n').filter(l => l.trim());

            if (posLines.length === 0 || negLines.length === 0) {
                throw new Error('Please provide at least one positive and one negative example');
            }

            const total = posLines.length + negLines.length;
            let current = 0;

            const posVectors: Float32Array[] = [];
            const negVectors: Float32Array[] = [];

            // Process positives
            for (const line of posLines) {
                console.log('VectorLab: Computing positive example:', line);
                const v = await onCompute(line);
                if (v) {
                    console.log('VectorLab: Got vector of length:', v.length);
                    posVectors.push(v);
                } else {
                    console.warn('VectorLab: Failed to get vector for:', line);
                }
                current++;
                setProgress(Math.round((current / total) * 100));
            }

            // Process negatives
            for (const line of negLines) {
                console.log('VectorLab: Computing negative example:', line);
                const v = await onCompute(line);
                if (v) {
                    console.log('VectorLab: Got vector of length:', v.length);
                    negVectors.push(v);
                } else {
                    console.warn('VectorLab: Failed to get vector for:', line);
                }
                current++;
                setProgress(Math.round((current / total) * 100));
            }

            console.log('VectorLab: Positive vectors:', posVectors.length, 'Negative vectors:', negVectors.length);

            if (posVectors.length === 0) {
                throw new Error('Failed to compute any positive examples. Check browser console for details.');
            }
            if (negVectors.length === 0) {
                throw new Error('Failed to compute any negative examples. Check browser console for details.');
            }

            const vector = computeSteeringVector(posVectors, negVectors);
            console.log('VectorLab: Steering vector computed, length:', vector.length);
            setComputedVector(vector);
            setVectorName(`${posLines[0]} vs ${negLines[0]}`);

        } catch (e) {
            console.error(e);
            await message(String(e), { title: 'Computation Failed', kind: 'error' });
        } finally {
            setIsComputing(false);
            setProgress(0);
        }
    };

    const handleSave = () => {
        if (!computedVector || !vectorName) return;
        const data = createSteeringVectorData(vectorName, computedVector);
        onVectorCreated(data);
        setComputedVector(null);
        setVectorName('');
    };

    return (
        <div className="vector-lab">
            <h3>Vector Lab (Steering)</h3>
            <p className="hint">Define a concept to steer the model towards or away from.</p>

            <div className="examples-grid">
                <div className="example-column">
                    <label>Positive Examples (+)</label>
                    <textarea
                        value={posExamples}
                        onChange={e => setPosExamples(e.target.value)}
                        rows={5}
                        placeholder="One phrase per line"
                    />
                </div>
                <div className="example-column">
                    <label>Negative Examples (-)</label>
                    <textarea
                        value={negExamples}
                        onChange={e => setNegExamples(e.target.value)}
                        rows={5}
                        placeholder="One phrase per line"
                    />
                </div>
            </div>

            <div className="controls">
                <button
                    onClick={handleCompute}
                    disabled={isComputing}
                    className="btn-primary"
                >
                    {isComputing ? `Computing... ${progress}%` : 'Compute Vector'}
                </button>
            </div>

            {computedVector && (
                <div className="save-form">
                    <h4>Vector Computed! ({computedVector.length} dim)</h4>
                    <div className="input-group">
                        <input
                            type="text"
                            value={vectorName}
                            onChange={e => setVectorName(e.target.value)}
                            placeholder="Vector Name"
                        />
                        <button onClick={handleSave} className="btn-success">Save Vector</button>
                    </div>
                </div>
            )}

            <div className="saved-list">
                <h4>Saved Vectors</h4>
                {savedVectors.length === 0 ? <p className="empty">No vectors saved</p> : (
                    <ul>
                        {savedVectors.map(v => (
                            <li key={v.id} className="saved-item">
                                <span className="vector-name">{v.name}</span>
                                <div className="vector-actions">
                                    <button onClick={() => onLoadVector(v)} className="btn-sm">Load</button>
                                    <button onClick={() => onDeleteVector(v.id)} className="btn-sm btn-danger">×</button>
                                </div>
                            </li>
                        ))}
                    </ul>
                )}
            </div>
        </div>
    );
}
