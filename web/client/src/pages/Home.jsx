import React, { useState } from "react";
import Gangastondog from "../assets/images/gangastondog.jpg";
import axios from "axios";

function Home() {
  const [prompt, setPrompt] = useState("");
  const [response, setResponse] = useState("");

  const handleSubmit = async (event) => {
    event.preventDefault();
    try {
      // Send the prompt to the Flask server
      const res = await axios.post("http://localhost:5000/api/prompt", {
        prompt: prompt,
      });

      // Set the response from the server
      setResponse(res.data.response);
    } catch (error) {
      console.error("Error sending prompt:", error);
      setResponse("Error: Unable to fetch response from the server.");
    }
  };

  return (
    <div className="flex min-h-[90vh]">
      {/* Left Content (Chat Container) */}
      <div className="w-3/5 p-4 bg-white border border-gray-300 rounded-lg shadow-md flex flex-col">
        {/* Chat Content */}
        <div className="flex flex-col overflow-y-auto">
          {/* Chat messages or any other content */}
          <div className="bg-gray-100 p-4 mb-4 w-[80%] max-w-none rounded text-base self-end font-inter">
            Hello why am i here? 
          </div>
          {/* response && here */}
          {(
            <div className="flex items-start gap-4 p-4 w-[80%]">
            {/* Bot Image  */}
            <div className="w-12 h-12 rounded-full bg-gray-300 flex items-center justify-center">
              {/* Replace with your bot image */}
              <img
                src={Gangastondog}
                alt="Bot Icon"
                className="w-12 h-12 rounded-full"
              />
            </div>

            {/* Bot Response Text */}
            <div className="bg-blue-500 p-4 rounded-lg border border-cyan-400 max-w-lg">
              <p className="text-white text-base font-inter">
                Hey there, I'm DoctorBot! Public Health Scotland have asked me to help fill out a medical form with you, would you like to begin?
              </p>
            </div>
          </div>

          )}
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="flex items-center gap-2 mt-auto">
          <textarea
            className="w-full p-2 rounded border border-gray-300 text-base"
            rows="2"
            placeholder="Type your question..."
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
          ></textarea>
          <button
            type="submit"
            className="bg-blue-500 text-white px-4 py-2 rounded cursor-pointer text-sm whitespace-nowrap hover:bg-blue-700"
          >
            Send
          </button>
        </form>
      </div>

      {/* Right Content (Extra Space) */}
      <div className="w-2/5 bg-gray-100 p-4">
        {/* Additional content here */}
        <p>Right-side content goes here...</p>
      </div>
    </div>
  );
}

export default Home;