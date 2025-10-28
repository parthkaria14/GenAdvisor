/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    // Add this line to include your ai-advisor-view.tsx
    "./ai-advisor-view.tsx", 
  ],
  theme: {
    extend: {},
  },
  plugins: [
    require('@tailwindcss/typography'), // This enables the 'prose' classes
  ],
}