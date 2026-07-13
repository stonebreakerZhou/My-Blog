export function initHeadingAnchors() {
	const headings = document.querySelectorAll('.prose :is(h2, h3, h4)[id]');
	for (const heading of headings) {
		if (heading.querySelector('.heading-anchor')) continue;
		const id = heading.id;

		const anchor = document.createElement('a');
		anchor.className = 'heading-anchor';
		anchor.href = `#${id}`;
		anchor.textContent = '#';
		anchor.setAttribute('aria-label', 'Link to this section');

		anchor.addEventListener('click', (event) => {
			event.preventDefault();
			heading.scrollIntoView({ behavior: 'smooth', block: 'start' });
			history.replaceState(null, '', `#${id}`);
			const url = `${location.origin}${location.pathname}#${id}`;
			navigator.clipboard?.writeText(url).catch(() => {});
			anchor.classList.add('is-copied');
			setTimeout(() => anchor.classList.remove('is-copied'), 1200);
		});

		heading.appendChild(anchor);
	}
}
