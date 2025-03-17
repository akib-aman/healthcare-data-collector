import React from "react";
import { Routes, Route} from "react-router-dom";
import Home from "./pages/Home";
import About from "./pages/About";
import Help from "./pages/Help";
import FormComplete from "./pages/FormComplete";
import Layout from "./layout";

function App() {
  return (
    <div>
      <Layout>
        <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/about" element={<About />} />
            <Route path="/help" element={<Help />} />
            <Route path="/FormComplete" element={<FormComplete />} />
        </Routes>
      </Layout>

    </div>
  );
}

export default App;
