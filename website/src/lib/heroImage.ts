import { getImage } from 'astro:assets';
import pappouHero from '../../public/assets/original-artworks/pappou_hero.png';

/** Widths the hero picture and its preload share. Mobile lab (412px at 1.75 dpr) selects 750. */
export const HERO_WIDTHS = [640, 750, 960, 1280, 1536] as const;
export const HERO_SIZES = '100vw';

export async function heroAvifSrcSet(): Promise<string> {
  const images = await Promise.all(
    HERO_WIDTHS.map((width) => getImage({ src: pappouHero, width, format: 'avif' })),
  );
  return images.map((image, index) => `${image.src} ${HERO_WIDTHS[index]}w`).join(', ');
}
