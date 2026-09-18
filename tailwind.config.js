/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "var(--bg)",
        "bg-soft": "var(--bg-soft)",
        panel: "var(--panel)",
        "panel-2": "var(--panel-2)",
        line: "var(--border)",
        "line-soft": "var(--border-soft)",
        fg: "var(--text)",
        "fg-2": "var(--text-2)",
        "fg-3": "var(--text-3)",
        acc: "var(--acc)",
        "acc-soft": "var(--acc-soft)",
        good: "var(--green)",
        warn: "var(--amber)",
        bad: "var(--red)",
        info: "var(--blue)",
      },
      borderRadius: {
        sm: "8px",
        DEFAULT: "10px",
        md: "10px",
        lg: "14px",
        xl: "18px",
      },
      fontFamily: {
        sans: [
          "Inter",
          "-apple-system",
          "BlinkMacSystemFont",
          "Segoe UI",
          "Roboto",
          "sans-serif",
        ],
        mono: [
          "JetBrains Mono",
          "SF Mono",
          "ui-monospace",
          "Cascadia Code",
          "Consolas",
          "monospace",
        ],
      },
    },
  },
  plugins: [],
};
