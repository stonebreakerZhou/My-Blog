import sharp from 'sharp';
import path from 'node:path';
import { statSync } from 'node:fs';

const SRC = 'E:/personal_website/note_drafts/images';
const DST = 'E:/personal_website/stonebreaker-blog/public/blog-images';

const TARGETS = [
    'Lec16_ICA_eg1_source.jpg',
    'Lec16_ICA_illustration_observe.jpg',
    'Lec16_ICA_illustration_source.jpg',
    'Lec16_PCA_eg1_dataset.jpg',
    'Lec16_PCA_intuition_illustration.jpg',
    'Lec16_PCA_pre-processed_dataset.jpg',
    'Lec16_unstable_eigenvectors.jpg',
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
    const out = (before/1024).toFixed(1);
    const aft2 = (after/1024).toFixed(1);
    console.log(`✓ ${name}  →  ${outName}  (${out}KB → ${aft2}KB, -${ratio}%)`);
}

console.log(`\nDone: ${TARGETS.length} images converted.`);