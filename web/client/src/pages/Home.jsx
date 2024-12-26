import React, { useState } from "react";
import axios from "axios";
import "./Home.css";

function Home() {
  const [prompt, setPrompt] = useState("");
  const [response, setResponse] = useState("");

  const handleSubmit = async (event) => {
    event.preventDefault();
    try {
      const res = await axios.post("http://localhost:5000/api/prompt", {
        prompt: prompt,
      });
      setResponse(res.data.response);
    } catch (error) {
      console.error("Error sending prompt:", error);
    }
  };

  return (
    <div className="main-container"> 
      <div className="chat-container"> 
        <form onSubmit={handleSubmit} className="prompt-form">
          <label htmlFor="prompt">Ask a question:</label>
          <textarea
            id="prompt"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={4}
            placeholder="Type your question..."
          />
          <button type="submit">Send</button>
        </form>
  
        {response && (
          <div className="response-box">
            <h2>AI Response:</h2>
            <p>{response}</p>
          </div>
        )}
      </div>
  
      <div className="form-container"> 
        {/* Your form content here */} 
      </div>
    </div>
  );
}

export default Home;