// src/Layout.js
import React from "react";
import "./global.css"; // your shared styles

function Layout({ children }) {
  return (
    <div>
      <header className="header">
        <h1>Public Health Scotland</h1>
      </header>

      {/* main content area */}
      <div className="page-container">
        {children}
      </div>

      <footer className="footer">
        <p>© 2024 Healthcare App</p>
      </footer>
    </div>
  );
}

export default Layout;
