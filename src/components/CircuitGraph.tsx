import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import type { CircuitInfo } from '../types';
import { getCircuitTypeName } from '../utils/circuitDetection';

interface CircuitGraphProps {
    circuits: CircuitInfo[];
}

export function CircuitGraph({ circuits }: CircuitGraphProps) {
    const svgRef = useRef<SVGSVGElement>(null);

    useEffect(() => {
        if (!svgRef.current || circuits.length === 0) return;

        const width = 400;
        const height = 300;
        const svg = d3.select(svgRef.current);
        svg.selectAll('*').remove();

        // Prepare nodes and links
        // Nodes are the circuits
        // Links? For MVP, just show nodes clustered by type or layer
        const nodes = circuits.map((c, i) => ({
            ...c,
            id: i,
            r: 5 + c.score * 10, // Size by score
        }));

        const simulation = d3.forceSimulation(nodes as any)
            .force('charge', d3.forceManyBody().strength(-20))
            .force('center', d3.forceCenter(width / 2, height / 2))
            .force('y', d3.forceY().y((d: any) => d.layer * 30).strength(0.5)) // Layer separation
            .on('tick', ticked);

        const g = svg.append('g');

        const node = g.selectAll('circle')
            .data(nodes)
            .enter()
            .append('circle')
            .attr('r', d => d.r)
            .attr('fill', d => d.type === 'induction' ? '#ff6b6b' : '#4ecdc4')
            .attr('stroke', '#fff')
            .attr('stroke-width', 1.5)
            .call(drag(simulation) as any);

        node.append('title')
            .text(d => `${getCircuitTypeName(d.type)} (L${d.layer}H${d.head})`);

        function ticked() {
            node
                .attr('cx', d => (d as any).x)
                .attr('cy', d => (d as any).y);
        }

        // Drag behavior
        function drag(simulation: any) {
            function dragstarted(event: any) {
                if (!event.active) simulation.alphaTarget(0.3).restart();
                event.subject.fx = event.subject.x;
                event.subject.fy = event.subject.y;
            }

            function dragged(event: any) {
                event.subject.fx = event.x;
                event.subject.fy = event.y;
            }

            function dragended(event: any) {
                if (!event.active) simulation.alphaTarget(0);
                event.subject.fx = null;
                event.subject.fy = null;
            }

            return d3.drag()
                .on('start', dragstarted)
                .on('drag', dragged)
                .on('end', dragended);
        }

    }, [circuits]);

    if (circuits.length === 0) return <div className="no-circuits">No circuits detected in this forward pass.</div>;

    return (
        <div className="circuit-graph">
            <h3>Detected Circuits Graph</h3>
            <svg ref={svgRef} width="100%" height="300" viewBox="0 0 400 300" />
            <div className="legend">
                <span style={{ color: '#ff6b6b' }}>● Induction</span>
                <span style={{ color: '#4ecdc4', marginLeft: '10px' }}>● Other</span>
            </div>
        </div>
    );
}
