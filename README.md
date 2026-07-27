# Stonebreaker Blog

AI undergraduate @ BIT, writing notes on ML, DL, RL, AI safety and the math behind them.

## ✍️ Writing a new post

This blog uses **Obsidian as editor + Astro as engine**, with no conversion in between.

Read [`WRITING.md`](../WRITING.md) (in the repo root) for the full workflow:

- Obsidian Vault setup
- File naming conventions
- Frontmatter schema
- Image / formula / code handling
- How to embed interactive demos

TL;DR — keep `_TEMPLATE.md` updated, write in Obsidian, push to git, done.

## 🧞 Commands

| Command                   | Action                                           |
| :------------------------ | :----------------------------------------------- |
| `npm install`             | Install dependencies                            |
| `npm run dev`             | Start local dev server at `localhost:4321`      |
| `npm run build`           | Build production site to `./dist/`              |
| `npm run preview`         | Preview the production build locally            |
| `npm run astro check`     | Type-check the project                          |
| `npm run astro -- --help` | Astro CLI help                                  |

## 🚀 Project Structure

```
src/
├── components/        Reusable Astro components
├── content/blog/      Markdown / MDX posts (this is the Obsidian Vault)
│   └── _TEMPLATE.md   Blank template — copy this for new posts
├── layouts/           Page layouts (BaseLayout, BlogPost)
├── pages/             Routes (index, about, posts)
├── scripts/           Client-side scripts (TOC, reading progress, code copy)
└── styles/            Global CSS
```

## 👀 Stack

- [Astro 6](https://astro.build/) — static site generator
- [MDX](https://mdxjs.com/) — Markdown + components
- [KaTeX](https://katex.org/) via `remark-math` + `rehype-katex` — math rendering
- [Three.js](https://threejs.org/) — for interactive demos

Based on the [Astro blog starter](https://github.com/withastro/astro/tree/main/examples/blog) / [Bear Blog](https://github.com/HermanMartinus/bearblog/).