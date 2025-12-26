/**
 * Sounio Playground - Share Functionality
 *
 * This module handles URL generation and parsing for sharing code snippets.
 * It uses LZ-string compression to keep URLs reasonably short.
 */

declare const LZString: {
    compressToEncodedURIComponent: (input: string) => string;
    decompressFromEncodedURIComponent: (input: string) => string | null;
    compressToBase64: (input: string) => string;
    decompressFromBase64: (input: string) => string | null;
};

/**
 * Share configuration
 */
interface ShareConfig {
    code: string;
    version?: string;
    theme?: 'dark' | 'light';
}

/**
 * Compress code using LZ-string
 */
function compressCode(code: string): string {
    if (typeof LZString !== 'undefined') {
        return LZString.compressToEncodedURIComponent(code);
    }
    // Fallback to base64 if LZ-string is not available
    return btoa(unescape(encodeURIComponent(code)));
}

/**
 * Decompress code using LZ-string
 */
function decompressCode(compressed: string): string | null {
    if (typeof LZString !== 'undefined') {
        return LZString.decompressFromEncodedURIComponent(compressed);
    }
    // Fallback from base64
    try {
        return decodeURIComponent(escape(atob(compressed)));
    } catch {
        return null;
    }
}

/**
 * Generate a shareable URL for the given code
 */
export function generateShareUrl(code: string, config?: Partial<ShareConfig>): string {
    const compressed = compressCode(code);

    const params = new URLSearchParams();
    params.set('code', compressed);

    if (config?.version) {
        params.set('v', config.version);
    }

    if (config?.theme) {
        params.set('theme', config.theme);
    }

    const baseUrl = window.location.origin + window.location.pathname;
    return `${baseUrl}?${params.toString()}`;
}

/**
 * Load code from the current URL
 */
export function loadFromUrl(): string | null {
    const params = new URLSearchParams(window.location.search);
    const compressed = params.get('code');

    if (!compressed) {
        return null;
    }

    const code = decompressCode(compressed);

    if (!code) {
        console.warn('Failed to decompress code from URL');
        return null;
    }

    // Apply theme from URL if present
    const theme = params.get('theme');
    if (theme === 'light' || theme === 'dark') {
        if (theme === 'light') {
            document.documentElement.classList.add('light');
        } else {
            document.documentElement.classList.remove('light');
        }
        localStorage.setItem('sounio-theme', theme);
    }

    return code;
}

/**
 * Create a GitHub Gist with the code
 * Note: Requires authentication token
 */
export async function createGist(
    code: string,
    options: {
        description?: string;
        filename?: string;
        isPublic?: boolean;
        token?: string;
    } = {}
): Promise<string | null> {
    const {
        description = 'Sounio Playground Code',
        filename = 'code.sounio',
        isPublic = true,
        token
    } = options;

    if (!token) {
        console.warn('GitHub token required for Gist creation');
        return null;
    }

    try {
        const response = await fetch('https://api.github.com/gists', {
            method: 'POST',
            headers: {
                'Accept': 'application/vnd.github+json',
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                description,
                public: isPublic,
                files: {
                    [filename]: {
                        content: code,
                    },
                },
            }),
        });

        if (!response.ok) {
            throw new Error(`Failed to create gist: ${response.status}`);
        }

        const gist = await response.json();
        return gist.html_url;
    } catch (error) {
        console.error('Failed to create gist:', error);
        return null;
    }
}

/**
 * Load code from a GitHub Gist URL
 */
export async function loadFromGist(gistUrl: string): Promise<string | null> {
    // Extract gist ID from URL
    const match = gistUrl.match(/gist\.github\.com\/[^/]+\/([a-f0-9]+)/i);
    if (!match) {
        console.warn('Invalid gist URL');
        return null;
    }

    const gistId = match[1];

    try {
        const response = await fetch(`https://api.github.com/gists/${gistId}`, {
            headers: {
                'Accept': 'application/vnd.github+json',
            },
        });

        if (!response.ok) {
            throw new Error(`Failed to fetch gist: ${response.status}`);
        }

        const gist = await response.json();

        // Find the first .sounio or .d file, or just the first file
        const files = Object.keys(gist.files);
        const sounioFile = files.find(f => f.endsWith('.sounio') || f.endsWith('.d'));
        const filename = sounioFile || files[0];

        if (!filename) {
            return null;
        }

        return gist.files[filename].content;
    } catch (error) {
        console.error('Failed to load gist:', error);
        return null;
    }
}

/**
 * Copy text to clipboard with fallback
 */
export async function copyToClipboard(text: string): Promise<boolean> {
    try {
        await navigator.clipboard.writeText(text);
        return true;
    } catch {
        // Fallback for older browsers
        const textarea = document.createElement('textarea');
        textarea.value = text;
        textarea.style.position = 'fixed';
        textarea.style.opacity = '0';
        document.body.appendChild(textarea);
        textarea.select();

        try {
            document.execCommand('copy');
            return true;
        } catch {
            return false;
        } finally {
            document.body.removeChild(textarea);
        }
    }
}

/**
 * Generate a short hash for code (for analytics or caching)
 */
export function hashCode(code: string): string {
    let hash = 0;
    for (let i = 0; i < code.length; i++) {
        const char = code.charCodeAt(i);
        hash = ((hash << 5) - hash) + char;
        hash = hash & hash; // Convert to 32bit integer
    }
    return Math.abs(hash).toString(36);
}

export default {
    generateShareUrl,
    loadFromUrl,
    createGist,
    loadFromGist,
    copyToClipboard,
    hashCode,
};
