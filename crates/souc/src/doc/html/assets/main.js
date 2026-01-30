// Demetrios Documentation - Main JavaScript

(function() {
    'use strict';

    // Theme toggle
    const themeToggle = document.getElementById('theme-toggle');
    const lightTheme = document.getElementById('theme-light');
    const darkTheme = document.getElementById('theme-dark');

    function setTheme(theme) {
        if (theme === 'dark') {
            lightTheme.disabled = true;
            darkTheme.disabled = false;
            document.body.setAttribute('data-theme', 'dark');
        } else {
            lightTheme.disabled = false;
            darkTheme.disabled = true;
            document.body.setAttribute('data-theme', 'light');
        }
        localStorage.setItem('theme', theme);
    }

    // Load saved theme
    const savedTheme = localStorage.getItem('theme') ||
        (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    setTheme(savedTheme);

    if (themeToggle) {
        themeToggle.addEventListener('click', function() {
            const current = localStorage.getItem('theme') || 'light';
            setTheme(current === 'light' ? 'dark' : 'light');
        });
    }

    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(function(anchor) {
        anchor.addEventListener('click', function(e) {
            const targetId = this.getAttribute('href').slice(1);
            const target = document.getElementById(targetId);
            if (target) {
                e.preventDefault();
                target.scrollIntoView({ behavior: 'smooth' });
                history.pushState(null, null, '#' + targetId);
            }
        });
    });

    // Highlight current section in sidebar
    function highlightCurrentSection() {
        const sections = document.querySelectorAll('h2[id], h3[id]');
        const navLinks = document.querySelectorAll('.sidebar-nav a');

        let current = '';
        sections.forEach(function(section) {
            const rect = section.getBoundingClientRect();
            if (rect.top <= 100) {
                current = section.id;
            }
        });

        navLinks.forEach(function(link) {
            const href = link.getAttribute('href');
            if (href && href.includes('#' + current)) {
                link.classList.add('active');
            } else {
                link.classList.remove('active');
            }
        });
    }

    window.addEventListener('scroll', highlightCurrentSection);
    highlightCurrentSection();

    // Copy code button
    document.querySelectorAll('pre').forEach(function(pre) {
        const button = document.createElement('button');
        button.className = 'copy-button';
        button.textContent = 'Copy';
        button.addEventListener('click', function() {
            const code = pre.querySelector('code');
            const text = code ? code.textContent : pre.textContent;
            navigator.clipboard.writeText(text).then(function() {
                button.textContent = 'Copied!';
                setTimeout(function() {
                    button.textContent = 'Copy';
                }, 2000);
            });
        });
        pre.style.position = 'relative';
        button.style.cssText = 'position:absolute;top:0.5rem;right:0.5rem;padding:0.25rem 0.5rem;font-size:0.75rem;cursor:pointer;border:1px solid var(--border-color);border-radius:var(--border-radius);background:var(--bg-color);color:var(--text-color);';
        pre.appendChild(button);
    });

    // Mobile sidebar toggle
    const sidebar = document.querySelector('.sidebar');
    const content = document.querySelector('.content');

    if (window.innerWidth <= 768 && sidebar) {
        const toggle = document.createElement('button');
        toggle.className = 'sidebar-toggle';
        toggle.textContent = 'Menu';
        toggle.style.cssText = 'position:fixed;bottom:1rem;right:1rem;z-index:100;padding:0.5rem 1rem;border:1px solid var(--border-color);border-radius:var(--border-radius);background:var(--bg-color);color:var(--text-color);cursor:pointer;';

        toggle.addEventListener('click', function() {
            sidebar.classList.toggle('open');
        });

        document.body.appendChild(toggle);
    }
})();
