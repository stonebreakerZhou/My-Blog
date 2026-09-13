import sharp from 'sharp';
import path from 'node:path';

const SRC = 'E:/personal_website/note_drafts/images';
const DST = 'E:/personal_website/stonebreaker-blog/public/blog-images';

const TARGETS = [
	'Lec5_GDA_approach.png',
	'Lec5_GDA_logistic_comparison.jpg',
	'Lec5_GDA_simple_example_figure1.jpg',
	'Lec5_GDA_simple_example_figure2.jpg',
	'Lec5_GDA_simple_example_figure3.jpg',
	'Lec5_LW_figure.png',
	'Lec5_multivariate_Guassian_bumps1.png',
	'Lec5_multivariate_Guassian_bumps2.png',
	'Lec5_multivariate_Guassian_bumps3.png',
	'Lec5_multivariate_Guassian_contours.png',
	'Lec5_naive_Bayes_prob_graph.jpg',
];

for (const name of TARGETS) {
	const srcPath = path.join(SRC, name);
	const outName = name.replace(/\.(png|jpg|jpeg)$/i, '.webp');
	const dstPath = path.join(DST, outName);
	const before = (await import('node:fs')).statSync(srcPath).size;
	const pipeline = sharp(srcPath).resize({ width: 1080, withoutEnlargement: true }).webp({ quality: 85 });
	const info = await pipeline.toFile(dstPath);
	const after = info.size;
	const ratio = ((1 - after / before) * 100).toFixed(1);
	console.log(`${name} → ${outName}  ${(before/1024).toFixed(0)}KB → ${(after/1024).toFixed(0)}KB  (-${ratio}%)  ${info.width}x${info.height}`);
}