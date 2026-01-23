import { useEffect, useState } from 'react';
import type { SteeringConfig } from '../types';
import type { SteeringVectorData } from '../utils/vectorMath';

interface SteeringPanelProps {
    config: SteeringConfig;
    onChange: (config: SteeringConfig) => void;
    activeVectorData: SteeringVectorData | null;
}

export function SteeringPanel({ config, onChange, activeVectorData }: SteeringPanelProps) {
    const [localStrength, setLocalStrength] = useState(config.strength);

    // Sync internal state with props
    useEffect(() => {
        setLocalStrength(config.strength);
    }, [config.strength]);

    const handleToggle = () => {
        onChange({ ...config, enabled: !config.enabled });
    };

    const handleStrengthChange = (val: number) => {
        setLocalStrength(val);
        onChange({ ...config, strength: val });
    };

    if (!activeVectorData && !config.vector) {
        return (
            <div className="steering-panel disabled">
                <h3>Steering Controller</h3>
                <p className="hint">Load a vector from the Vector Lab to enable steering.</p>
            </div>
        );
    }

    return (
        <div className={`steering-panel ${config.enabled ? 'active' : ''}`}>
            <div className="panel-header">
                <h3>Steering Controller</h3>
                <label className="switch">
                    <input
                        type="checkbox"
                        checked={config.enabled}
                        onChange={handleToggle}
                    />
                    <span className="slider round"></span>
                </label>
            </div>

            <div className="active-vector-card">
                <div className="vector-icon">⚡</div>
                <div className="vector-details">
                    <span className="vector-label">Active Vector</span>
                    <span className="vector-title">{activeVectorData?.name || config.vectorName || 'Custom Vector'}</span>
                    <span className="vector-meta">{activeVectorData?.dimension || config.vector?.length} dim</span>
                </div>
            </div>

            <div className="strength-control">
                <div className="strength-header">
                    <span className="label">Injection Strength</span>
                    <span className="value">{localStrength.toFixed(1)}x</span>
                </div>
                <input
                    type="range"
                    min="-5"
                    max="5"
                    step="0.1"
                    value={localStrength}
                    onChange={e => handleStrengthChange(Number(e.target.value))}
                    className="range-input"
                />
                <div className="range-marks">
                    <span>-5</span>
                    <span className="zero">0</span>
                    <span>+5</span>
                </div>

                <div className="quick-actions">
                    <button onClick={() => handleStrengthChange(localStrength * -1)}>Flip</button>
                    <button onClick={() => handleStrengthChange(0)}>Reset</button>
                </div>
            </div>
        </div>
    );
}
