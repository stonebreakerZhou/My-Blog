import sharp from 'sharp';
import path from 'node:path';
import { statSync } from 'node:fs';

const SRC = 'E:/personal_website/note_drafts/images';
const DST = 'E:/personal_website/stonebreaker-blog/public/blog-images';

const TARGETS = [
	'Lec9_data_view.jpg',
	'Lec9_ERM.jpg',
	'Lec9_error_risk_plot.jpg',
	'Lec9_hypothesis_space.jpg',
	'Lec9_learning_algo_procedure.jpg',
	'Lec9_parameter_view.jpg',
	'Lec9_epsilon(hat(h))_with_epsilon(h).jpg',
	'Lec9_epsilon(hat(h))_with_epsilon(h^star).jpg',
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