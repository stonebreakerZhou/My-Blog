import sharp from 'sharp';
import path from 'node:path';
import { statSync } from 'node:fs';

const SRC = 'E:/personal_website/note_drafts/images';
const DST = 'E:/personal_website/stonebreaker-blog/public/blog-images';

const TARGETS = [
	'Lec10_boosting_dataset.jpg',
	'Lec10_boosting_decision-boundary1.jpg',
	'Lec10_boosting_decision-boundary1_mistakes.jpg',
	'Lec10_boosting_decision-boundary2.jpg',
	'Lec10_CE-loss_curve_midpoint_eg.jpg',
	'Lec10_CE-loss_curve1.jpg',
	'Lec10_CE-loss_curve2.jpg',
	'Lec10_CE-loss_curve3.jpg',
	'Lec10_DT_dataset_1.jpg',
	'Lec10_DT_dataset_1_split.jpg',
	'Lec10_DT_dataset3.jpg',
	'Lec10_DT_depth.jpg',
	'Lec10_DT_eg1.jpg',
	'Lec10_DT_no-additive.jpg',
	'Lec10_Gini-loss_curve.jpg',
	'Lec10_loss-change_on_CE-loss_curve.jpg',
	'Lec10_misclassification_simple-eg1_dataset.jpg',
	'Lec10_misclassification_simple-eg1_split1.jpg',
	'Lec10_misclassification_simple-eg1_split2.jpg',
	'Lec10_misclassification-loss_curve.jpg',
	'Lec10_regression-tree_dataset.jpg',
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