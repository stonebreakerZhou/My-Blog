let observer = null;

export function initTocActive() {
	if (observer) {
		observer.disconnect();
		observer = null;
	}

	const links = document.querySelectorAll('[data-toc-link]');
	if (links.length === 0) return;

	// If the user has the TOC folded, skip observer setup entirely.
	if (document.documentElement.getAttribute('data-toc') === 'closed') {
		for (const link of links) link.classList.remove('is-active');
		return;
	}

	const linkFor = new Map();
	for (const link of links) {
		const id = link.getAttribute('data-toc-link');
		linkFor.set(id, link);

		if (!link.dataset.tocBound) {
			link.dataset.tocBound = '1';
			link.addEventListener('click', (event) => {
				const target = document.getElementById(id);
				if (!target) return;
				event.preventDefault();
				target.scrollIntoView({ behavior: 'smooth', block: 'start' });
				history.replaceState(null, '', `#${id}`);
			});
		}
	}

	const headings = document.querySelectorAll('.prose :is(h2, h3)[id]');
	if (headings.length === 0) return;

	const visible = new Set();

	function setActive(id) {
		for (const link of links) link.classList.remove('is-active');
		linkFor.get(id)?.classList.add('is-active');
	}

	observer = new IntersectionObserver(
		(entries) => {
			for (const entry of entries) {
				if (entry.isIntersecting) visible.add(entry.target.id);
				else visible.delete(entry.target.id);
			}
			// Highlight the first heading (in document order) currently in view.
			for (const h of headings) {
				if (visible.has(h.id)) {
					setActive(h.id);
					return;
				}
			}
		},
		{ rootMargin: '0px 0px -70% 0px', threshold: 0 },
	);

	for (const h of headings) observer.observe(h);
}
