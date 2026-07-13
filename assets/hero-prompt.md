# Hero 背景图生成提示词

**目标尺寸**：1920×720（约 2.67:1 横向 banner；移动端会被竖向裁切，重要元素放中间 60%）

**目标风格**：极简艺术插画风（minimalist illustrated art）—— 不是写实摄影、不是水彩、不是油画。参考：Studio Ghibli 的简笔夜空、a16z 播客封面、Beeple 早期极简作品、The Atlantic 文章头图。

---

## 主提示词

```
A minimalist illustrated art banner of a serene night sky, soft
hand-painted textures, deep teal and muted indigo gradient with
subtle plum undertones, gentle hand-drawn star clusters scattered
across the sky like ink dots, varying sizes and opacities, the
word "STONEBREAKER" is written across the center as a celestial
constellation where each letter is formed by 5 to 9 hand-painted
dots of slightly varying size, weight and brightness, the dots
are connected by faint dashed hand-drawn lines like a vintage
astronomer's star chart, the letters feel organic and slightly
uneven, never printed or typed, generous negative space above and
below the text, soft lavender and pale gold nebula washes behind
the letters, a few larger glowing stars with subtle 4-point
starbursts, occasional thin hand-drawn streak of a shooting
star, the overall mood is calm contemplative and intimate rather
than dramatic, painted with gouache or digital illustration
technique, soft brush textures, no hard edges, no photography,
no photorealism, ultra wide 8:3 aspect ratio, 2K illustration
```

## 负面提示词

```
photorealistic, photograph, astrophotography, milky way, galaxy,
moon, planets, sun, comet, satellite, printed text, typography,
font, glyph, geometric shapes, sharp edges, hard lines, neon
colors, oversaturated, cyberpunk, sci-fi, anime, cartoon, chibi,
3D render, CGI, frame, border, logo, watermark, signature, harsh
contrast, dramatic lighting, lens flare from camera, people,
face, hand, text artifact, jpeg artifacts
```

## 为什么这么写

**"绘画感"如何具体化**：
- 关键词：`hand-painted`, `hand-drawn`, `gouache`, `digital illustration`, `soft brush textures`
- 显式排除：`photorealistic`, `photograph`, `astrophotography`
- 不要写 `oil painting`（太厚重） / `watercolor`（太软化）—— `gouache`（水粉）正好：不透明但有手作感

**"字母自然"如何具体化**：
- `5 to 9 hand-painted dots` — 字母是**点**不是**字形**
- `varying size, weight and brightness` — **每颗点不一样**
- `slightly uneven` — **故意不工整**（这一句是反 AI 过度规整的关键）
- `vintage astronomer's star chart` — 借经典星图的视觉语言（手绘、虚线）
- 显式排除：`printed text`, `typography`, `font`, `glyph`

**"博客调性"如何具体化**：
- `calm contemplative and intimate rather than dramatic` — 直接给 AI 定情绪基调
- `muted indigo`, `subtle plum undertones`, `pale gold` — 颜色饱和度都压低
- `generous negative space` — 留白是博客图最重要的特征
- 显式排除：`neon`, `oversaturated`, `cyberpunk`, `sci-fi`, `dramatic lighting`

---

## Midjourney 参数

```
--ar 8:3 --style raw --stylize 100 --niji 6
```
- `--niji 6` 走**插画模型**而不是写实模型（关键！默认 `--v 6` 是写实）
- `--style raw` 减少风格化覆盖
- `--stylize 100` 偏克制（默认 100，750 偏艺术、夸张）

如果用 `--niji` 字母太"萌"了，换 `--v 6 --style raw --stylize 50`，更克制。

## DALL-E 3 参数

直接粘主提示词。**关键**：在 prompt 末尾加一句：
```
Do not add any text or typography to the image. The word
STONEBREAKER must be made entirely of illustrated dots and
hand-drawn lines, not letters.
```
（DALL-E 经常忍不住加印刷字）

## Stable Diffusion

- 模型：**AnythingV5**（插画风） / **DreamShaper**（写实偏插画）
- 采样器 `DPM++ 2M Karras`，步数 28，CFG 6.5
- 加 LoRA：**illustration-style / flat-color** 系列
- 负面 prompt 把上面的负面词也喂进去

## 如果字母还是不"自然"

追加这一句再跑：
```
the constellation STONEBREAKER should look like a child's
constellation drawing on dark blue paper, the dots are
unevenly placed, the lines are wobbly and hand-drawn, not
perfectly straight
```
（"孩子画的星座"是反 AI 完美主义最有效的 prompt）

## 落地

- 推荐格式：`.webp`（体积小）或 `.png`（无损）
- 保存路径：`public/hero-bg.webp`
- 替换：把 `src/pages/index.astro` 里 `.hero-banner` 的 `background` 渐变整段删掉，换成：
  ```css
  background-image: url('/hero-bg.webp');
  background-size: cover;
  background-position: center;
  ```
- 然后 hero 里那个 SVG 星座 (`<svg class="hero-constellation">`) 整段删掉——AI 生成的图本身就有字母了
- 渐变背景（深蓝紫色）也删掉，让真实星空透出来

## 验收清单

打开图看：
- [ ] 字母 **STONEBREAKER** 看上去是**手绘的星点**（不是字形 / 字体）
- [ ] 字母**大小、间距、亮度不完美**（这就是"自然"）
- [ ] 整体风格是**绘画/插画**（不是照片）
- [ ] 颜色**不刺眼**（饱和度低、留白多）
- [ ] 字母**可读**——不需要想就能认
- [ ] 画面**没有 AI 文字伪影**（边缘不会多出"STONEBREAKERER"之类的字）
