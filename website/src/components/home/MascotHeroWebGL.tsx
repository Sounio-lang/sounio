import { createSignal, onMount, onCleanup } from 'solid-js';
import * as THREE from 'three';

export default function MascotHeroWebGL() {
  let containerRef: HTMLDivElement | undefined;
  const [activeSide, setActiveSide] = createSignal<'left' | 'right' | null>(null);

  onMount(() => {
    if (!containerRef) return;

    // Basic Three.js setup
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    containerRef.appendChild(renderer.domElement);

    camera.position.z = 5;

    // Particles
    const particlesGeometry = new THREE.BufferGeometry();
    const particlesCount = 5000;
    const posArray = new Float32Array(particlesCount * 3);

    for(let i = 0; i < particlesCount * 3; i++) {
      posArray[i] = (Math.random() - 0.5) * 10;
    }

    particlesGeometry.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
    
    // Custom Shader Material for "Uncertainty" effect
    const particlesMaterial = new THREE.ShaderMaterial({
      uniforms: {
        uTime: { value: 0 },
        uMouseX: { value: 0 },
        uMouseY: { value: 0 },
        uResolution: { value: new THREE.Vector2(window.innerWidth, window.innerHeight) }
      },
      vertexShader: `
        uniform float uTime;
        uniform float uMouseX;
        void main() {
          vec3 pos = position;
          // Add noise/uncertainty based on mouse position
          float noise = sin(pos.x * 10.0 + uTime) * 0.1;
          pos.y += noise * (1.0 - abs(uMouseX)); 
          
          vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
          gl_PointSize = 2.0 * (1.0 / -mvPosition.z);
          gl_Position = projectionMatrix * mvPosition;
        }
      `,
      fragmentShader: `
        void main() {
          // Gold/Teal mix
          gl_FragColor = vec4(0.84, 0.7, 0.35, 0.6); 
        }
      `,
      transparent: true,
      blending: THREE.AdditiveBlending
    });

    const particlesMesh = new THREE.Points(particlesGeometry, particlesMaterial);
    scene.add(particlesMesh);

    // Mouse tracking
    let mouseX = 0;
    let mouseY = 0;

    const handleMouseMove = (event: MouseEvent) => {
      mouseX = (event.clientX / window.innerWidth) * 2 - 1;
      mouseY = -(event.clientY / window.innerHeight) * 2 + 1;
      
      particlesMaterial.uniforms.uMouseX.value = mouseX;
      particlesMaterial.uniforms.uMouseY.value = mouseY;

      if (mouseX < -0.1) setActiveSide('left');
      else if (mouseX > 0.1) setActiveSide('right');
      else setActiveSide(null);
    };

    window.addEventListener('mousemove', handleMouseMove);

    // Animation loop
    const clock = new THREE.Clock();

    const animate = () => {
      const elapsedTime = clock.getElapsedTime();
      particlesMaterial.uniforms.uTime.value = elapsedTime;

      // Gentle rotation
      particlesMesh.rotation.y = mouseX * 0.1;
      particlesMesh.rotation.x = -mouseY * 0.1;

      renderer.render(scene, camera);
      requestAnimationFrame(animate);
    };

    animate();

    // Resize handler
    const handleResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
      particlesMaterial.uniforms.uResolution.value.set(window.innerWidth, window.innerHeight);
    };

    window.addEventListener('resize', handleResize);

    onCleanup(() => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('resize', handleResize);
      if (containerRef && renderer.domElement) {
        containerRef.removeChild(renderer.domElement);
      }
      particlesGeometry.dispose();
      particlesMaterial.dispose();
      renderer.dispose();
    });
  });

  return (
    <div class="relative w-full h-screen overflow-hidden bg-[#020617]">
      <div ref={containerRef} class="absolute inset-0 z-0" />
      
      <div class="relative z-10 flex w-full h-full pointer-events-none">
        {/* Left Side - Yiayia */}
        <div class={`flex-1 flex flex-col justify-center items-center transition-opacity duration-700 ${activeSide() === 'right' ? 'opacity-10' : 'opacity-100'}`}>
          <div class="pointer-events-auto cursor-pointer group" onClick={() => window.location.href = '/start?guide=yiayia'}>
            <h2 class="font-display text-6xl md:text-8xl text-white font-medium tracking-tight group-hover:text-[var(--color-accent-teal)] transition-colors">
              Measure Lives.
            </h2>
            <p class="text-[var(--color-text-secondary)] mt-4 text-xl opacity-0 group-hover:opacity-100 transition-opacity">
              Start with Yiayia →
            </p>
          </div>
        </div>

        {/* Right Side - Pappou */}
        <div class={`flex-1 flex flex-col justify-center items-center transition-opacity duration-700 ${activeSide() === 'left' ? 'opacity-10' : 'opacity-100'}`}>
          <div class="pointer-events-auto cursor-pointer group" onClick={() => window.location.href = '/start?guide=pappou'}>
            <h2 class="font-display text-6xl md:text-8xl text-white font-medium tracking-tight group-hover:text-[var(--color-accent-gold)] transition-colors">
              Measure Systems.
            </h2>
            <p class="text-[var(--color-text-secondary)] mt-4 text-xl opacity-0 group-hover:opacity-100 transition-opacity">
              Start with Pappou →
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}