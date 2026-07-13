const STORAGE_KEY = 'toc';
const STATE_OPEN = 'open';
const STATE_CLOSED = 'closed';

function readState() {
	try {
		return localStorage.getItem(STORAGE_KEY) === STATE_CLOSED
			? STATE_CLOSED
			: STATE_OPEN;
	} catch {
		return STATE_OPEN;
	}
}

function writeState(state) {
	try {
		localStorage.setItem(STORAGE_KEY, state);
	} catch {}
}

function applyState(state) {
	document.documentElement.setAttribute('data-toc', state);
}

export function initTocToggle() {
	const area = document.querySelector('.toc-area');
	const toggle = document.querySelector('.toc-toggle');
	const close = document.querySelector('.toc-close');
	if (!area || !toggle) return;

	// Sync DOM with whatever state was applied at boot (BaseHead no-flash).
	const current = document.documentElement.getAttribute('data-toc') ?? STATE_OPEN;
	applyState(current);

	const flip = () => {
		const next =
			document.documentElement.getAttribute('data-toc') === STATE_CLOSED
				? STATE_OPEN
				: STATE_CLOSED;
		applyState(next);
		writeState(next);
	};

	toggle.addEventListener('click', flip);
	close?.addEventListener('click', flip);
}
