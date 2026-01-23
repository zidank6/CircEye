// Check if running in Tauri environment
const isTauri = () => {
    return typeof window !== 'undefined' && '__TAURI__' in window;
};

// Browser-based download fallback
function downloadBlob(blob: Blob, filename: string) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Export SVG element as file
export async function exportAsSvg(svgElement: SVGSVGElement): Promise<boolean> {
    try {
        const serializer = new XMLSerializer();
        const svgString = serializer.serializeToString(svgElement);
        const fullSvg = `<?xml version="1.0" encoding="UTF-8"?>\n${svgString}`;

        if (isTauri()) {
            // Use Tauri APIs
            const { save } = await import('@tauri-apps/plugin-dialog');
            const { writeFile } = await import('@tauri-apps/plugin-fs');

            const path = await save({
                filters: [{ name: 'SVG', extensions: ['svg'] }],
                defaultPath: 'attention-visualization.svg',
            });

            if (!path) return false;
            await writeFile(path, new TextEncoder().encode(fullSvg));
        } else {
            // Browser fallback
            const blob = new Blob([fullSvg], { type: 'image/svg+xml' });
            downloadBlob(blob, 'attention-visualization.svg');
        }

        return true;
    } catch (e) {
        console.error('SVG export failed:', e);
        return false;
    }
}

// Export SVG element as PNG
export async function exportAsPng(
    svgElement: SVGSVGElement,
    width = 1200,
    height = 800
): Promise<boolean> {
    try {
        const serializer = new XMLSerializer();
        let svgString = serializer.serializeToString(svgElement);

        // Ensure SVG has proper dimensions and namespace for rendering
        if (!svgString.includes('xmlns=')) {
            svgString = svgString.replace('<svg', '<svg xmlns="http://www.w3.org/2000/svg"');
        }

        // Create canvas for PNG conversion
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');

        if (!ctx) throw new Error('Canvas context unavailable');

        // Create image from SVG
        const img = new Image();
        const blob = new Blob([svgString], { type: 'image/svg+xml;charset=utf-8' });
        const url = URL.createObjectURL(blob);

        await new Promise<void>((resolve, reject) => {
            img.onload = () => resolve();
            img.onerror = (e) => {
                console.error('Image load error:', e);
                reject(new Error('Failed to load SVG as image'));
            };
            img.src = url;
        });

        // Draw to canvas
        ctx.fillStyle = '#0d1117';
        ctx.fillRect(0, 0, width, height);
        ctx.drawImage(img, 0, 0, width, height);

        URL.revokeObjectURL(url);

        // Convert to PNG blob
        const pngBlob = await new Promise<Blob | null>(resolve =>
            canvas.toBlob(resolve, 'image/png')
        );

        if (!pngBlob) throw new Error('PNG conversion failed');

        if (isTauri()) {
            // Use Tauri APIs
            const { save } = await import('@tauri-apps/plugin-dialog');
            const { writeFile } = await import('@tauri-apps/plugin-fs');

            const path = await save({
                filters: [{ name: 'PNG', extensions: ['png'] }],
                defaultPath: 'attention-visualization.png',
            });

            if (!path) return false;

            const arrayBuffer = await pngBlob.arrayBuffer();
            await writeFile(path, new Uint8Array(arrayBuffer));
        } else {
            // Browser fallback
            downloadBlob(pngBlob, 'attention-visualization.png');
        }

        return true;
    } catch (e) {
        console.error('PNG export failed:', e);
        return false;
    }
}
