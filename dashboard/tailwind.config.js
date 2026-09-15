/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // Dark clinical theme.
        surface: {
          DEFAULT: '#0b1220',
          raised: '#111a2e',
          border: '#1e2a44',
        },
        accent: {
          DEFAULT: '#22d3ee',
          soft: '#67e8f9',
        },
        signal: {
          normal: '#34d399',
          abnormal: '#f87171',
          warn: '#fbbf24',
        },
      },
      fontFamily: {
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
    },
  },
  plugins: [],
};
