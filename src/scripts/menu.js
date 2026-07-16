function syncAndBind() {
	const menu = document.querySelector('.menu');
	if (!(menu instanceof HTMLButtonElement)) return;
	if (!menu.dataset.bound) {
		menu.dataset.bound = 'true';
		menu.addEventListener('click', () => {
			const isExpanded = menu.getAttribute('aria-expanded') === 'true';
			menu.setAttribute('aria-expanded', `${!isExpanded}`);
		});
	}
	// After a View Transitions swap, force the menu closed so the
	// :has() selector in Header.astro hides the dropdown on the new page.
	menu.setAttribute('aria-expanded', 'false');
}

syncAndBind();
document.addEventListener('astro:after-swap', syncAndBind);