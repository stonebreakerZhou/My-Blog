let rafId = null;
let onScroll = null;

export function initReadingProgress() {
	const bar = document.querySelector('.reading-progress');
	const article = document.querySelector('.article');
	if (!bar || !article) return;

	function update() {
		rafId = null;
		const rect = article.getBoundingClientRect();
		const total = rect.height - window.innerHeight;
		const scrolled = -rect.top;
		const progress = total > 0 ? Math.min(1, Math.max(0, scrolled / total)) : 0;
		bar.style.transform = `scaleX(${progress})`;
	}

	if (onScroll) {
		window.removeEventListener('scroll', onScroll);
		window.removeEventListener('resize', onScroll);
	}
	onScroll = () => {
		if (rafId == null) rafId = requestAnimationFrame(update);
	};
	window.addEventListener('scroll', onScroll, { passive: true });
	window.addEventListener('resize', onScroll, { passive: true });
	update();
}
