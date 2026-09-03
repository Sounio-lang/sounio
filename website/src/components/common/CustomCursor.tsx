import { useEffect, useState } from 'react';
import { motion, useSpring } from 'framer-motion';

export function CustomCursor() {
  const [isHovering, setIsHovering] = useState(false);
  const [isVisible, setIsVisible] = useState(false);
  const cursorX = useSpring(0, { stiffness: 600, damping: 30, mass: 0.5 });
  const cursorY = useSpring(0, { stiffness: 600, damping: 30, mass: 0.5 });

  useEffect(() => {
    // Hide on mobile / touch devices or screens smaller than 1024px to prevent stuck cursors on tap
    if (typeof window !== 'undefined' && (window.matchMedia("(hover: none)").matches || window.innerWidth < 1024)) {
      return;
    }

    // Respect OS “reduce motion” — keep native cursor semantics
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
    if (reduceMotion.matches) {
      return;
    }

    const moveCursor = (e: MouseEvent) => {
      cursorX.set(e.clientX);
      cursorY.set(e.clientY);
      if (!isVisible) setIsVisible(true);

      const target = e.target as HTMLElement;
      if (target.closest('a') || target.closest('button') || target.closest('input')) {
        setIsHovering(true);
      } else {
        setIsHovering(false);
      }
    };

    const handleMouseLeave = () => setIsVisible(false);
    const handleMouseEnter = () => setIsVisible(true);

    window.addEventListener('mousemove', moveCursor);
    document.addEventListener('mouseleave', handleMouseLeave);
    document.addEventListener('mouseenter', handleMouseEnter);

    return () => {
      window.removeEventListener('mousemove', moveCursor);
      document.removeEventListener('mouseleave', handleMouseLeave);
      document.removeEventListener('mouseenter', handleMouseEnter);
    };
  }, [cursorX, cursorY, isVisible]);

  if (!isVisible) return null;

  return (
    <motion.div
      className="fixed top-0 left-0 w-6 h-6 rounded-full border border-white/50 pointer-events-none z-[9999] mix-blend-difference flex items-center justify-center"
      style={{
        translateX: "-50%",
        translateY: "-50%",
        x: cursorX,
        y: cursorY,
      }}
      animate={{
        scale: isHovering ? 2.5 : 1,
        backgroundColor: isHovering ? 'rgba(255,255,255,1)' : 'rgba(255,255,255,0)',
        borderWidth: isHovering ? '0px' : '1px'
      }}
      transition={{ duration: 0.15 }}
    />
  );
}
