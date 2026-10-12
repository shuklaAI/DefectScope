/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        ink: {
          950: "#f4f3ef",
          900: "#eeede8",
          800: "#ffffff",
          700: "#e8e7e1",
          600: "#d1d1ca",
        },
        accent: {
          500: "#8a4f24",
          400: "#75421f",
          600: "#693b1b",
        },
        warn: {
          500: "#a54232",
        },
        ok: {
          500: "#39714d",
        },
        slate: {
          100: "#242923",
          200: "#303630",
          300: "#414841",
          400: "#596159",
          500: "#727970",
          600: "#858b82",
          700: "#9ba097",
          800: "#b3b7af",
          900: "#c9ccc5",
        },
      },
      fontFamily: {
        mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "monospace"],
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
      },
    },
  },
  plugins: [],
};
