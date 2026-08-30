import sharp from 'sharp';
import path from 'node:path';
import { statSync } from 'node:fs';

const SRC = 'E:/personal_website/note_drafts/images';
const DST = 'E:/personal_website/stonebreaker-blog/public/blog-images';

const TARGETS = [
	'Lec7_w_perpendicular_boundary.jpg',
	'Lec7_w-boudary_eg1.jpg',
	'Lec7_w-boudary_eg2.jpg',
	'Lec7_Phi_dot-product.jpg',
	'Lec7_not-linearly-separable-data.jpg',
	'Lec7_soft_SVM_boudary.jpg',
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
	console.log(`${name} → ${outName}  ${(before/1024).toFixed(0)}KB → ${(after/1024).toFixed(0)}KB  (-${ratio}%)  ${info.width}x${info.height}`);
}
