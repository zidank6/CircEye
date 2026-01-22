import React, { useRef, useEffect, useState } from 'react';
import * as d3 from 'd3';
import type { AttentionData } from '../types';

interface AttentionVizProps {
    attentions: number[][][][] | null;
    tokens: string[];
    numLayers: number;
    numHeads: number;
    onSvgRef?: (ref: SVGSVGElement | null) => void;
}

export function AttentionViz({
    attentions,
    tokens,
    numLayers,
    numHeads,
    onSvgRef,
}: AttentionVizProps) {
    const svgRef = useRef<SVGSVGElement>(null);
    const [selectedLayer, setSelectedLayer] = useState(0);
    const [selectedHead, setSelectedHead] = useState(0);
    const [hoveredCell, setHoveredCell] = useState<{ i: number; j: number; value: number } | null>(null);

    useEffect(() => {
        if (onSvgRef) onSvgRef(svgRef.current);
    }, [onSvgRef]);

    useEffect(() => {
        if (!attentions || !svgRef.current || tokens.length === 0) return;

        const svg = d3.select(svgRef.current);
        svg.selectAll('*').remove();

        const matrix = attentions[selectedLayer]?.[selectedHead];
        if (!matrix) return;

        const margin = { top: 60, right: 20, bottom: 20, left: 80 };
        const width = 500 - margin.left - margin.right;
        const height = 500 - margin.top - margin.bottom;

        const seqLen = Math.min(matrix.length, tokens.length);
        const cellSize = Math.min(width, height) / seqLen;

        const g = svg
            .attr('width', width + margin.left + margin.right)
            .attr('height', height + margin.top + margin.bottom)
            .append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);

        // Color scale from white to blue
        const colorScale = d3.scaleSequential(d3.interpolateBlues)
            .domain([0, 1]);

        // Draw heatmap cells
        for (let i = 0; i < seqLen; i++) {
            for (let j = 0; j < seqLen; j++) {
                const value = matrix[i][j];
                g.append('rect')
                    .attr('x', j * cellSize)
                    .attr('y', i * cellSize)
                    .attr('width', cellSize - 1)
                    .attr('height', cellSize - 1)
                    .attr('fill', colorScale(value))
                    .attr('stroke', '#333')
                    .attr('stroke-width', 0.5)
                    .style('cursor', 'pointer')
                    .on('mouseover', () => setHoveredCell({ i, j, value }))
                    .on('mouseout', () => setHoveredCell(null));
            }
        }

        // X-axis labels (query tokens)
        const truncate = (s: string) => s.length > 6 ? s.slice(0, 5) + '…' : s;

        g.selectAll('.x-label')
            .data(tokens.slice(0, seqLen))
            .enter()
            .append('text')
            .attr('class', 'x-label')
            .attr('x', (_, i) => i * cellSize + cellSize / 2)
            .attr('y', -5)
            .attr('text-anchor', 'middle')
            .attr('font-size', '10px')
            .attr('fill', '#e0e0e0')
            .text(d => truncate(d));

        // Y-axis labels (key tokens)
        g.selectAll('.y-label')
            .data(tokens.slice(0, seqLen))
            .enter()
            .append('text')
            .attr('class', 'y-label')
            .attr('x', -5)
            .attr('y', (_, i) => i * cellSize + cellSize / 2)
            .attr('text-anchor', 'end')
            .attr('alignment-baseline', 'middle')
            .attr('font-size', '10px')
            .attr('fill', '#e0e0e0')
            .text(d => truncate(d));

        // Highlight hovered cell
        if (hoveredCell) {
            g.append('rect')
                .attr('x', hoveredCell.j * cellSize)
                .attr('y', hoveredCell.i * cellSize)
                .attr('width', cellSize - 1)
                .attr('height', cellSize - 1)
                .attr('fill', 'none')
                .attr('stroke', '#ffcc00')
                .attr('stroke-width', 2);
        }
    }, [attentions, tokens, selectedLayer, selectedHead, hoveredCell]);

    return (
        <div className="attention-viz">
            <div className="viz-controls">
                <label>
                    Layer:
                    <select
                        value={selectedLayer}
                        onChange={e => setSelectedLayer(Number(e.target.value))}
                    >
                        {Array.from({ length: numLayers }, (_, i) => (
                            <option key={i} value={i}>{i}</option>
                        ))}
                    </select>
                </label>
                <label>
                    Head:
                    <select
                        value={selectedHead}
                        onChange={e => setSelectedHead(Number(e.target.value))}
                    >
                        {Array.from({ length: numHeads }, (_, i) => (
                            <option key={i} value={i}>{i}</option>
                        ))}
                    </select>
                </label>
            </div>

            <svg ref={svgRef} className="viz-svg" />

            {hoveredCell && (
                <div className="viz-tooltip">
                    Attending from "{tokens[hoveredCell.i]}" to "{tokens[hoveredCell.j]}": {hoveredCell.value.toFixed(4)}
                </div>
            )}
        </div>
    );
}
