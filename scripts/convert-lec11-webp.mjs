import sharp from 'sharp';
import path from 'node:path';
import { statSync } from 'node:fs';

const SRC = 'E:/personal_website/note_drafts/images';
const DST = 'E:/personal_website/stonebreaker-blog/public/blog-images';

const TARGETS = [
    'Lec11_LR_eg1.jpg',
    'Lec11_LR_eg2.jpg',
    'Lec11_softmax_eg.jpg',
    'Lec11_neural_network_architecture_eg.jpg',
    'Lec11_neural_network_parameters.jpg',
    'Lec11_neural_network_3layers.jpg',
    'Lec11_house_price_neural_network_with-knowledge.jpg',
    'Lec11_house_price_neural_network_fully-connected.jpg',
    'Lec11_propagation_architecture.jpg',
    'Lec11_batched_input.jpg',
    'Lec11_batched_linear_part.jpg',
    'Lec11_ReLU.jpg',
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