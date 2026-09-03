import { useRef, useEffect } from 'react';
import { useScroll } from 'framer-motion';

export function EpistemicFluidCanvas() {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  // Track scroll progress to transition from Chaos -> Order
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start end", "end start"]
  });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d', { alpha: false });
    if (!ctx) return;

    // Static backdrop instead of continuous particle animation.
    if (typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      const paintStill = () => {
        const rect = canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        canvas.width = Math.max(1, Math.floor(rect.width * dpr));
        canvas.height = Math.max(1, Math.floor(rect.height * dpr));
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.scale(dpr, dpr);
        const gradient = ctx.createLinearGradient(0, 0, rect.width, rect.height);
        gradient.addColorStop(0, '#020617');
        gradient.addColorStop(0.48, '#071f36');
        gradient.addColorStop(1, '#0b1020');
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, rect.width, rect.height);

        const nodes = [
          [0.18, 0.3, 34, 'rgba(214, 179, 90, 0.2)'],
          [0.42, 0.62, 50, 'rgba(64, 196, 255, 0.16)'],
          [0.72, 0.38, 42, 'rgba(238, 214, 145, 0.18)'],
          [0.84, 0.74, 28, 'rgba(125, 211, 252, 0.14)'],
        ] as const;

        for (const [x, y, radius, color] of nodes) {
          ctx.beginPath();
          ctx.arc(rect.width * x, rect.height * y, radius, 0, Math.PI * 2);
          ctx.fillStyle = color;
          ctx.fill();
        }
      };
      paintStill();
      window.addEventListener('resize', paintStill);
      return () => window.removeEventListener('resize', paintStill);
    }

    let animationFrameId: number;
    let isPlaying = false;
    let width = 0;
    let height = 0;
    let orderProgress = 0;

    const unsubscribe = scrollYProgress.on('change', (v) => {
      orderProgress = v; // 0 to 1
    });

    const particles: any[] = [];
    const numParticles = typeof window !== 'undefined' && window.innerWidth < 768 ? 400 : 1200;

    const initParticles = () => {
      particles.length = 0;
      for (let i = 0; i < numParticles; i++) {
        particles.push({
          x: Math.random() * width,
          y: Math.random() * height,
          vx: (Math.random() - 0.5) * 3,
          vy: (Math.random() - 0.5) * 3,
          targetX: 0,
          targetY: 0,
          color: `hsla(${180 + Math.random() * 50}, 100%, ${60 + Math.random() * 20}%, ${0.4 + Math.random() * 0.6})`,
          size: 0.8 + Math.random() * 2,
          noiseOffset: Math.random() * 1000
        });
      }
      
      // Assign lattice targets in a centered circular or grid pattern
      const cols = Math.ceil(Math.sqrt(numParticles * (width / height)));
      const rows = Math.ceil(numParticles / cols);
      const cellW = width / cols;
      const cellH = height / rows;
      
      let count = 0;
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          if (count < numParticles) {
            // Add a slight wave/curve distortion to the lattice
            const tx = c * cellW + cellW / 2;
            const ty = r * cellH + cellH / 2;
            const dx = tx - width / 2;
            const dy = ty - height / 2;
            const dist = Math.sqrt(dx * dx + dy * dy);
            const angle = Math.atan2(dy, dx) + dist * 0.002;
            
            particles[count].targetX = width / 2 + Math.cos(angle) * dist;
            particles[count].targetY = height / 2 + Math.sin(angle) * dist;
            count++;
          }
        }
      }
    };

    const resize = () => {
      const parent = canvas.parentElement;
      if (parent) {
        width = parent.clientWidth;
        height = parent.clientHeight;
        canvas.width = width * window.devicePixelRatio;
        canvas.height = height * window.devicePixelRatio;
        ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
        initParticles();
      }
    };

    window.addEventListener('resize', resize);
    resize();

    let time = 0;
    const render = () => {
      if (!isPlaying) return;
      time += 0.01;
      
      // Trail effect
      ctx.fillStyle = 'rgba(2, 6, 23, 0.15)'; // Sounio background color #020617
      ctx.fillRect(0, 0, width, height);

      // Scroll range 0.2 to 0.7 maps to order 0 to 1
      const effectiveOrder = Math.max(0, Math.min(1, (orderProgress - 0.2) * 2));

      particles.forEach(p => {
        if (effectiveOrder < 0.01) {
          // Pure chaos state
          p.x += p.vx;
          p.y += p.vy;
          
          // Gentle pull to center to avoid them all leaving the screen
          p.vx += (width / 2 - p.x) * 0.0001;
          p.vy += (height / 2 - p.y) * 0.0001;

          if (p.x < 0 || p.x > width) p.vx *= -0.8;
          if (p.y < 0 || p.y > height) p.vy *= -0.8;
        } else {
          // Transition to ordered lattice state
          const dx = p.targetX - p.x;
          const dy = p.targetY - p.y;
          
          // Calculate noise based on time
          const noiseX = Math.sin(time + p.noiseOffset) * 2;
          const noiseY = Math.cos(time + p.noiseOffset) * 2;
          
          // The closer to order 1, the stronger the pull to target
          const pullStrength = 0.03 + (0.07 * effectiveOrder);
          
          p.vx = p.vx * 0.9 + dx * pullStrength;
          p.vy = p.vy * 0.9 + dy * pullStrength;

          // Add chaos noise scaled inversely by order
          const chaosMultiplier = (1 - effectiveOrder) * 4;
          
          p.x += (p.vx * effectiveOrder) + (noiseX * chaosMultiplier);
          p.y += (p.vy * effectiveOrder) + (noiseY * chaosMultiplier);
        }

        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fillStyle = p.color;
        ctx.fill();
      });

      animationFrameId = requestAnimationFrame(render);
    };

    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          if (!isPlaying) {
            isPlaying = true;
            render();
          }
        } else {
          isPlaying = false;
          if (animationFrameId) cancelAnimationFrame(animationFrameId);
        }
      });
    }, { threshold: 0 });

    observer.observe(canvas);

    return () => {
      unsubscribe();
      window.removeEventListener('resize', resize);
      observer.disconnect();
      if (animationFrameId) cancelAnimationFrame(animationFrameId);
    };
  }, []);

  return (
    <section ref={containerRef} className="relative h-[250vh] bg-[#020617] text-white">
      <div className="sticky top-0 h-screen w-full flex items-center justify-center overflow-hidden">
        {/* The canvas itself */}
        <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" />
        
        {/* Text Overlay */}
        <div className="relative z-10 text-center max-w-4xl px-6 p-8 rounded-3xl bg-black/20 backdrop-blur-sm border border-white/5 shadow-2xl">
          <h2 className="text-[clamp(2.5rem,5vw,4.5rem)] font-extrabold tracking-tighter mb-4 text-white leading-tight">
            From Chaos to <br className="md:hidden" /><span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-[var(--color-accent-teal)]">Rigorous Bounds.</span>
          </h2>
          <p className="text-[clamp(1.1rem,2vw,1.5rem)] text-white/70 font-light mx-auto max-w-[40ch]">
            Scroll to observe the epistemic state transition. Entropy collapses as formal proofs establish absolute mathematical certainty.
          </p>
        </div>
      </div>
    </section>
  );
}
