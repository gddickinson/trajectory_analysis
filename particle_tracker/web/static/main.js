// Entry point: lazy page modules, the mode label, a subtle background of blinking, drifting spots, then the router.
import { $, app, isStatic, lazy, route, start } from './core.js';

const PAGES = {
  'home.js': ['home'], 'analyze.js': ['analyze'], 'tracks.js': ['tracks'], 'features.js': ['features'],
  'motion.js': ['motion'], 'density.js': ['density'], 'lengthcheck.js': ['lengthcheck'], 'about.js': ['method', 'about'],
};
lazy(Object.fromEntries(Object.entries(PAGES).flatMap(([file, names]) => names.map(n => [n, () => import('./' + file)]))));
route('notfound', () => { app().innerHTML = '<h1>Not found</h1><p><a href="#/">Back to the start</a></p>'; });

$('#mode').textContent = isStatic()
  ? 'static site: the analysis runs in your browser (JS port tested against the Python code); files never leave your computer'
  : 'local app: the analysis runs in Python on this computer';

// Background: faint spots, some wandering, some trapped, leaving short trails. Off under reduced motion.
function background(canvas) {
  if (!canvas || matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  const g = canvas.getContext('2d');
  const fit = () => { canvas.width = innerWidth; canvas.height = innerHeight; };
  fit(); addEventListener('resize', fit);
  const cols = ['#4fd1ff', '#ff7a59', '#c58cff'];
  const ps = Array.from({ length: 20 }, (_, i) => {
    const x = Math.random() * innerWidth, y = Math.random() * innerHeight;
    return { x, y, cx: x, cy: y, kind: i % 3 };
  });
  let last = 0;
  const tick = t => {
    requestAnimationFrame(tick);
    if (document.hidden || t - last < 60) return;
    last = t;
    g.fillStyle = 'rgba(5,6,12,.07)'; g.fillRect(0, 0, canvas.width, canvas.height);
    for (const p of ps) {
      p.x += (Math.random() - 0.5) * (p.kind === 1 ? 1.5 : 3.5);
      p.y += (Math.random() - 0.5) * (p.kind === 1 ? 1.5 : 3.5);
      if (p.kind === 1) { p.x += (p.cx - p.x) * 0.1; p.y += (p.cy - p.y) * 0.1; }
      p.x = (p.x + canvas.width) % canvas.width; p.y = (p.y + canvas.height) % canvas.height;
      const r = g.createRadialGradient(p.x, p.y, 0, p.x, p.y, 5);
      r.addColorStop(0, cols[p.kind]); r.addColorStop(1, 'transparent');
      g.fillStyle = r; g.fillRect(p.x - 5, p.y - 5, 10, 10);
    }
  };
  requestAnimationFrame(tick);
}
background($('#bg'));
start();
