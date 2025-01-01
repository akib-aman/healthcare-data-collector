// src/Layout.js
import React from "react";

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

          <div className="bg-violet-[#3f3685] h-full flex-1 flex items-center justify-center  bg-[#3f3685] hover:bg-[#483d97]">
              {/* Insert a button here */}

            <span className="font-inter text-white font-semibold text-[1rem]"> 
              Home 
            </span>
              
          </div> 

          <div className="bg-violet-[#3f3685] h-full flex-1 flex items-center justify-center  bg-[#3f3685] hover:bg-[#483d97]">
              {/* Insert a button here */}

            <span className="font-inter text-white font-semibold text-[1rem]"> 
              Help 
            </span>
              
          </div> 

          <div className="bg-violet-[#3f3685] h-full flex-1 flex items-center justify-center  bg-[#3f3685] hover:bg-[#483d97]">
              {/* Insert a button here */}

            <span className="font-inter text-white font-semibold text-[1rem]"> 
              About 
            </span>
              
          </div> 

        </div>

      </header>

      {/* main content area */}
      <div className="page-container">
        {children}
      </div>

      <footer className="footer">
        <p className="w-full big-gray-200 text-center py-4">© 2024 Healthcare App</p>
      </footer>
    </div>
  );
}

export default Layout;
