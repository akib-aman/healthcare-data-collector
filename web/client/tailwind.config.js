/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}", // Add your JSX/TSX files here
  ],
  theme: {
    extend: {
      fontFamily: {
        inter : ['Inter', 'sans-serif'],
      },
    },
  },
  plugins: [],
};
