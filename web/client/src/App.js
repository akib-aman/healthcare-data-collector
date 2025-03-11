import React from "react";
import { Routes, Route} from "react-router-dom";
import Home from "./pages/Home";
import About from "./pages/About";
import Help from "./pages/Help";
import Layout from "./layout";

function App() {
  return (
    <div>
      <Layout>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/about" element={<About />} />
        <Route path="/help" element={<Help />} />
      </Routes>
    </Layout>

    </div>
  );
}

export default App;
