// Place any global data in this file.
// You can import this data from anywhere in your site by using the `import` keyword.

export const SITE_TITLE = 'Stonebreaker Blog';
export const SITE_DESCRIPTION = 'Break a stone every day';

// ============================================
// giscus 评论系统配置
// ============================================
// 数据全存 GitHub Discussions，无后端，适配 GitHub Pages 静态站。
// 调整这些值即可换仓库 / 换分类 / 加系列，不必改 layout 代码。

export const GISCUS_REPO = 'stonebreakerZhou/My-Blog';
export const GISCUS_REPO_ID = 'R_kgDOSVhbuw';
export const GISCUS_CATEGORY_ID = 'DIC_kwDOSVhbu84DGcBz';
// strict=1 → 只匹配标题中"包含"该字符串的 Discussion，锚点帖标题须包含 term。
// 以后为新课程开放评论：往这张表里加一行，并在对应分类下建一个标题含 term 的锚点帖。
export const GISCUS_TERM_BY_SERIES: Record<string, string> = {
	cs229: 'cs229',
};
export const GISCUS_TERM_FALLBACK = 'general';
