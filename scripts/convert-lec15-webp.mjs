import sharp from 'sharp';
import path from 'node:path';
import { statSync } from 'node:fs';

const SRC = 'E:/personal_website/note_drafts/images';
const DST = 'E:/personal_website/stonebreaker-blog/public/blog-images';

const TARGETS = [
    'Lec15_actual_sample_x.jpg',
    'Lec15_GMM_fit_eg.jpg',
    'Lec15_GMM_fit_problem_eg.jpg',
    'Lec15_Guassian_sample-z.jpg',
    'Lec15_x_without_noise.jpg',
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