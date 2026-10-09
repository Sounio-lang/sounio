import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import './styles/tokens.css';
import { App } from './App';

// Troca de tema: escreve data-theme na raiz e lembra a escolha.
// Sem escolha, o sistema decide — que é o estado da maioria dos visitantes.
const KEY = 'sounio-theme';
try {
  const saved = localStorage.getItem(KEY);
  if (saved === 'dark' || saved === 'light') {
    document.documentElement.setAttribute('data-theme', saved);
  }
} catch { /* modo privado, navegador sem storage: o sistema decide */ }

document.addEventListener('click', (e) => {
  const t = (e.target as HTMLElement)?.closest('[data-theme-toggle]');
  if (!t) return;
  const root = document.documentElement;
  const current = root.getAttribute('data-theme')
    ?? (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  const next = current === 'dark' ? 'light' : 'dark';
  root.setAttribute('data-theme', next);
  try { localStorage.setItem(KEY, next); } catch { /* idem */ }
});

createRoot(document.getElementById('root')!).render(
  <StrictMode><App /></StrictMode>
);
