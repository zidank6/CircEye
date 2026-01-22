import { save } from '@tauri-apps/plugin-dialog';
import { writeFile } from '@tauri-apps/plugin-fs';

// Export SVG element as file
export async function exportAsSvg(svgElement: SVGSVGElement): Promise<boolean> {
    try {
        const serializer = new XMLSerializer();
        const svgString = serializer.serializeToString(svgElement);

        // Add XML declaration and proper namespace
        const fullSvg = `<?xml version="1.0" encoding="UTF-8"?>\n${svgString}`;

        const path = await save({
            filters: [{ name: 'SVG', extensions: ['svg'] }],
            defaultPath: 'attention-visualization.svg',
        });

        if (!path) return false;

        await writeFile(path, new TextEncoder().encode(fullSvg));
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
        const svgString = serializer.serializeToString(svgElement);

        // Create canvas for PNG conversion
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');

        if (!ctx) throw new Error('Canvas context unavailable');

        // Create image from SVG
        const img = new Image();
        const blob = new Blob([svgString], { type: 'image/svg+xml' });
        const url = URL.createObjectURL(blob);

        await new Promise<void>((resolve, reject) => {
            img.onload = () => resolve();
            img.onerror = reject;
            img.src = url;
        });

        // Draw to canvas
        ctx.fillStyle = '#1a1a2e';
        ctx.fillRect(0, 0, width, height);
        ctx.drawImage(img, 0, 0, width, height);

        URL.revokeObjectURL(url);

        // Convert to PNG blob
        const pngBlob = await new Promise<Blob | null>(resolve =>
            canvas.toBlob(resolve, 'image/png')
        );

        if (!pngBlob) throw new Error('PNG conversion failed');

        const path = await save({
            filters: [{ name: 'PNG', extensions: ['png'] }],
            defaultPath: 'attention-visualization.png',
        });

        if (!path) return false;

        const arrayBuffer = await pngBlob.arrayBuffer();
        await writeFile(path, new Uint8Array(arrayBuffer));

        return true;
    } catch (e) {
        console.error('PNG export failed:', e);
        return false;
    }
}
