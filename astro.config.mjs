// @ts-check

import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import { defineConfig } from 'astro/config';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// rehype-katex's published types omit `this: Processor`, so strict TS
// rejects the [plugin, options] tuple. Runtime is fine — astro check passes.
const mathPlugins = /** @type {any} */ ({
	remarkPlugins: [remarkMath],
	rehypePlugins: [[rehypeKatex, { output: 'html' }]],
});

// https://astro.build/config
export default defineConfig({
	site: 'https://stonebreakerZhou.github.io',
	base: '/My-Blog',
	trailingSlash: 'always',
	integrations: [mdx(mathPlugins), sitemap()],
	markdown: mathPlugins,
});
