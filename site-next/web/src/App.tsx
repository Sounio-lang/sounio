import { useEffect, useState } from 'react';
import { Home } from './pages/Home';
import { Proof } from './pages/Proof';
import { Honesty } from './pages/Honesty';
import './App.css';

/** Roteamento por hash — sem dependência, e o site é estático de qualquer forma. */
function useRoute() {
  const read = () => (location.hash.replace(/^#\/?/, '') || 'home');
  const [route, setRoute] = useState(read);
  useEffect(() => {
    const on = () => { setRoute(read()); window.scrollTo(0, 0); };
    addEventListener('hashchange', on);
    return () => removeEventListener('hashchange', on);
  }, []);
  return route;
}

export function App() {
  const route = useRoute();

  return (
    <>
      <header className="masthead">
        <a className="lockup" href="#/">
          <img className="column" src="/brand/column_verdigris.png" alt="" aria-hidden="true" />
          <span className="wordmark">Sounio</span>
        </a>
        <nav className="nav">
          <a href="#/" aria-current={route === 'home' ? 'page' : undefined}>Home</a>
          <a href="#/honesty" aria-current={route === 'honesty' ? 'page' : undefined}>The argument</a>
          <a href="#/proof" aria-current={route === 'proof' ? 'page' : undefined}>Proof</a>
        </nav>
        <button className="theme" type="button" data-theme-toggle aria-label="Alternar tema">
          <span aria-hidden="true">◐</span>
        </button>
      </header>

      {route === 'proof' ? <Proof />
        : route === 'honesty' ? <Honesty />
        : <Home />}

      <footer className="colophon">
        <img className="column column-sm" src="/brand/column_verdigris.png" alt="" aria-hidden="true" />
        <p>
          Every numeral on this site came from <code>artifacts/**/*.json</code> via{' '}
          <code>claim()</code>. A hand-written number fails the build.
        </p>
      </footer>
    </>
  );
}
