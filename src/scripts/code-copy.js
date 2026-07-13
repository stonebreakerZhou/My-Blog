export function initCodeCopy() {
	const blocks = document.querySelectorAll('.prose pre');
	for (const pre of blocks) {
		if (pre.parentElement?.classList.contains('code-block')) continue;

		const wrapper = document.createElement('div');
		wrapper.className = 'code-block';
		pre.replaceWith(wrapper);
		wrapper.appendChild(pre);

		const button = document.createElement('button');
		button.type = 'button';
		button.className = 'code-copy';
		button.textContent = 'Copy';
		button.setAttribute('aria-label', 'Copy code to clipboard');

		button.addEventListener('click', async () => {
			try {
				await navigator.clipboard.writeText(pre.textContent ?? '');
				button.textContent = 'Copied!';
				button.classList.add('is-copied');
			} catch {
				button.textContent = 'Failed';
			}
			setTimeout(() => {
				button.textContent = 'Copy';
				button.classList.remove('is-copied');
			}, 1500);
		});

		wrapper.appendChild(button);
	}
}
