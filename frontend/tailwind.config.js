/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './app/**/*.{js,ts,jsx,tsx}',
    './components/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        // Map to CSS variables from globals.css
        primary: {
          50: '#e0f9ff',
          100: '#b8eeff',
          200: '#80e7ff',
          300: '#33d4ff',
          400: '#00c8ff',
          500: '#00c8ff',
          600: '#00a8cc',
          700: '#008099',
          800: '#005966',
          900: '#002e33',
        },
        dark: {
          900: '#050c15', // --bg
          800: '#0b1627', // --surface
          700: '#101e30', // --card
          600: '#132033', // --card2
        },
        accent: {
          DEFAULT: '#00c8ff',
          dim: 'rgba(0, 200, 255, 0.12)',
          glow: 'rgba(0, 200, 255, 0.25)',
        },
        success: '#00e5a0',
        danger: '#ff4a6a',
        warning: '#ffb800',
        muted: '#5a7a9a',
      },
      backgroundColor: {
        'surface-950': 'var(--bg)',
        'surface-900': 'var(--surface)',
        'surface-800': 'var(--card)',
        'surface-700': 'var(--card2)',
      },
      textColor: {
        'on-bg': 'var(--text)',
        'subtle': 'var(--muted)',
      },
      backgroundColor: {
        'gradient-radial': 'radial-gradient',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'spin-slow': 'spin 3s linear infinite',
        'bounce-slow': 'bounce 2s infinite',
        'glow-pulse': 'glow-pulse 4s ease-in-out infinite',
      },
      keyframes: {
        'rotate-meter': {
          '0%': { transform: 'rotate(0deg)' },
          '100%': { transform: 'rotate(360deg)' },
        },
        'glow-pulse': {
          '0%, 100%': { transform: 'scale(1)', opacity: '0.6' },
          '50%': { transform: 'scale(1.1)', opacity: '1' },
        },
      },
    },
  },
  plugins: [],
};
