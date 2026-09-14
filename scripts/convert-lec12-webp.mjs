import sharp from 'sharp';
import path from 'node:path';
import { statSync } from 'node:fs';

const SRC = 'E:/personal_website/note_drafts/images';
const DST = 'E:/personal_website/stonebreaker-blog/public/blog-images';

const TARGETS = [
    'Lec12_cost_function_compare.jpg',
    'Lec12_mini-batch_GD_plot.jpg',
    'Lec12_momentum_loss-plot1.jpg',
    'Lec12_momentum_loss-plot2.jpg',
    'Lec12_momentum_loss-plot3.jpg',
    'Lec12_momentum_loss-plot4.jpg',
    'Lec12_normalized_input.jpg',
    'Lec12_normalized_loss-plot_compare.jpg',
    'Lec12_normalized_loss-plot-route_compare.jpg',
    'Lec12_one_neuron_eg.jpg',
    'Lec12_raw_input.jpg',
    'Lec12_ReLU_plot.png',
    'Lec12_sigmoid_plot.jpg',
    'Lec12_simple-nn_eg.jpg',
    'Lec12_tanh_plot.png',
];

for (const name of TARGETS) {
    const srcPath = path.join(SRC, name);
    const outName = name.replace(/\.(png|jpg|jpeg)$/i, '.webp');
    const dstPath = path.join(DST, outName);
    const before = statSync(srcPath).size;
    const pipeline = sharp(srcPath)
        .resize({ width: 1080, withoutEnlargement: true })
        .webp({ quality: 85 });
    const info = await pipeline.toFile(dstPath);
    const after = info.size;
    const ratio = ((1 - after / before) * 100).toFixed(1);
    console.log(`✓ ${name}  →  ${outName}  (${(before/1024).toFixed(1)}KB → ${(after/1024).toFixed(1)}KB, -${ratio}%)`);
}

console.log(`\nDone: ${TARGETS.length} images converted.`);