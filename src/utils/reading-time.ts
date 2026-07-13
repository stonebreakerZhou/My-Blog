/**
 * Reading-time estimator that handles mixed Chinese + English content.
 *
 * CJK characters are counted individually (each ~ 1 "word" for reading-time math).
 * English words are split on whitespace / punctuation boundaries.
 * Default WPM of 250 covers mixed CN/EN prose at a comfortable reading pace.
 */

const CJK_RE = /[㐀-鿿가-힯]/g;
const EN_WORD_RE = /[a-zA-Z]+(?:['-][a-zA-Z]+)*/g;

export interface ReadingStats {
	words: number;
	minutes: number;
	cjkChars: number;
	enWords: number;
}

export function readingStats(text: string, wpm = 250): ReadingStats {
	const cjkChars = text.match(CJK_RE)?.length ?? 0;
	const enWords = text.match(EN_WORD_RE)?.length ?? 0;
	const words = cjkChars + enWords;
	const minutes = Math.max(1, Math.round(words / wpm));
	return { words, minutes, cjkChars, enWords };
}

export function formatReadingTime(stats: ReadingStats): string {
	return `${stats.minutes} min read`;
}

export function formatWordCount(stats: ReadingStats): string {
	return `${stats.words.toLocaleString('en-US')} ${stats.words === 1 ? 'word' : 'words'}`;
}
