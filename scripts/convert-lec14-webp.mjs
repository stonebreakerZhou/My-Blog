import sharp from 'sharp';
import path from 'node:path';
import { statSync } from 'node:fs';

const SRC = 'E:/personal_website/note_drafts/images';
const DST = 'E:/personal_website/stonebreaker-blog/public/blog-images';

const TARGETS = [
    'Lec14_EM_1.jpg',
    'Lec14_EM_2.jpg',
    'Lec14_EM_3.jpg',
    'Lec14_EM_4.jpg',
    'Lec14_EM_5.jpg',
    'Lec14_GMM_1.jpg',
    'Lec14_GMM_2.jpg',
    'Lec14_GMM_3.jpg',
    'Lec14_Jensen-inequality.jpg',
    'Lec14_k-means_1.jpg',
    'Lec14_k-means_2.jpg',
    'Lec14_k-means_3.jpg',
    'Lec14_k-means_4.jpg',
    'Lec14_k-means_5.jpg',
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