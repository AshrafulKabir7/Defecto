import { StrictMode, useEffect, useState, type ComponentType } from 'react';
import { createRoot } from 'react-dom/client';
import DemoPage from './pages/DemoPage';
import EvaluationPage from './pages/EvaluationPage';
import FaqPage from './pages/FaqPage';
import InspectPage from './pages/InspectPage';
import PilotPage from './pages/PilotPage';
import PricingPage from './pages/PricingPage';
import ProductPage from './pages/ProductPage';
import './styles.css';

const ROUTES: Record<string, { label: string; title: string; Page: ComponentType }> = {
  product: { label: 'Product', title: 'How it works', Page: ProductPage },
  demo: { label: 'Demo', title: 'Inspection demo', Page: DemoPage },
  inspect: { label: 'Live inspect', title: 'Live inspection', Page: InspectPage },
  evaluation: { label: 'Evaluation', title: 'Evaluation report', Page: EvaluationPage },
  pricing: { label: 'Pricing', title: 'Pricing', Page: PricingPage },
  faq: { label: 'FAQ', title: 'FAQ', Page: FaqPage },
  pilot: { label: 'Book a Pilot', title: 'Book a pilot', Page: PilotPage },
};
const NAV = ['product', 'demo', 'inspect', 'evaluation', 'pricing', 'faq'];

function currentRoute() {
  const key = window.location.hash.replace(/^#\/?/, '').split('?')[0];
  return key in ROUTES ? key : 'demo';
}

function Logo() {
  return (
    <svg viewBox="0 0 26 25" width="26" height="25" fill="none" aria-hidden="true">
      <circle cx="11.2" cy="11" r="8.6" stroke="#38c6ec" strokeWidth="3.2" />
      <circle cx="11.2" cy="11" r="2.9" fill="#38c6ec" />
      <path d="M17.6 17.4 L23.6 23.2" stroke="#38c6ec" strokeWidth="3.4" strokeLinecap="round" />
    </svg>
  );
}

function Nav({ route }: { route: string }) {
  const [open, setOpen] = useState(false);
  useEffect(() => setOpen(false), [route]);
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setOpen(false);
    const onClick = (e: MouseEvent) => !(e.target as HTMLElement).closest('.nav') && setOpen(false);
    window.addEventListener('keydown', onKey);
    document.addEventListener('click', onClick);
    return () => {
      window.removeEventListener('keydown', onKey);
      document.removeEventListener('click', onClick);
    };
  }, [open]);

  return (
    <header className="topbar">
      <nav className={`nav${open ? ' is-open' : ''}`} aria-label="Main">
        <a className="nav__brand" href="./">
          <Logo />
          <span>
            Defecto<sup>AI</sup>
          </span>
        </a>
        <div className="nav__links">
          {NAV.map((key) => (
            <a key={key} href={`#/${key}`} className={`nav__link${route === key ? ' is-active' : ''}`} aria-current={route === key ? 'page' : undefined}>
              {ROUTES[key].label}
            </a>
          ))}
        </div>
        <a className="nav__cta" href="#/pilot">
          Book a Pilot
        </a>
        <button
          type="button"
          className="burger"
          aria-label={open ? 'Close menu' : 'Open menu'}
          aria-expanded={open}
          aria-controls="app-menu"
          onClick={(e) => {
            e.stopPropagation();
            setOpen((o) => !o);
          }}
        >
          <span />
          <span />
        </button>
        <div className="navmenu" id="app-menu">
          {NAV.map((key) => (
            <a key={key} href={`#/${key}`} className={route === key ? 'is-active' : ''}>
              {ROUTES[key].label}
            </a>
          ))}
          <a className="cta" href="#/pilot">
            Book a Pilot
          </a>
        </div>
      </nav>
    </header>
  );
}

function App() {
  const [route, setRoute] = useState(currentRoute);
  useEffect(() => {
    const onHash = () => {
      setRoute(currentRoute());
      window.scrollTo(0, 0);
    };
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);
  useEffect(() => {
    document.title = `${ROUTES[route].title} | Defecto`;
  }, [route]);

  const { Page } = ROUTES[route];
  return (
    <>
      <Nav route={route} />
      <main>
        <Page key={route} />
      </main>
      <footer className="footer">
        <div className="footer__inner">
          <span>Defecto portfolio demo. Images: VisA dataset (Amazon Science, CC BY 4.0).</span>
          <span>
            <a href="./">Home</a> · <a href="#/evaluation">Evaluation</a> · <a href="#/pilot">Book a pilot</a>
          </span>
        </div>
      </footer>
    </>
  );
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
