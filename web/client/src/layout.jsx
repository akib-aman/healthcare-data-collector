// src/Layout.js
import React from "react";
import { Link } from "react-router-dom";

function Layout({ children }) {
  return (
    <div>
      <header>
        <div className="bg-[#5545a8] px-8 pb-6 pt-8">
          <h1 className="font-inter text-3xl font-bold text-white">
            Public Health Scotland
          </h1>
        </div>

        <div className="w-full h-12 bg-[#3f3685] flex justify-between">
          {/* Home Button */}
          <div className="flex-1 h-full bg-[#3f3685] hover:bg-[#483d97]">
            <Link
              to="/"
              className="no-underline w-full h-full flex items-center justify-center"
            >
              <span className="font-inter text-white font-semibold text-[1rem]">
                Home
              </span>
            </Link>
          </div>

          {/* Help Button */}
          <div className="flex-1 h-full bg-[#3f3685] hover:bg-[#483d97]">
            <Link
              to="/help"
              className="no-underline w-full h-full flex items-center justify-center"
            >
              <span className="font-inter text-white font-semibold text-[1rem]">
                Help
              </span>
            </Link>
          </div>

          {/* About Button */}
          <div className="flex-1 h-full bg-[#3f3685] hover:bg-[#483d97]">
            <Link
              to="/about"
              className="no-underline w-full h-full flex items-center justify-center"
            >
              <span className="font-inter text-white font-semibold text-[1rem]">
                About
              </span>
            </Link>
          </div>
        </div>
      </header>

      {/* main content area */}
      <div className="page-container">
        {children}
      </div>

      <footer className="footer">
        <p className="w-full big-gray-200 text-center py-4">
          © 2025 Healthcare App
        </p>
      </footer>
    </div>
  );
}

export default Layout;
