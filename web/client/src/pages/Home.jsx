import React, { useState } from "react";
import Gangastondog from "../assets/images/gangastondog.jpg";
import axios from "axios";

function Home() {
  const [prompt, setPrompt] = useState("");
  const [messages, setMessages] = useState([
    {
      type: "bot",
      text: "Hello! I'm DoctorBot. I'm here to help you fill out a medical form for Public Health Scotland. Let's begin! What is your age?",
    },
  ]);

  const handleSubmit = async (event) => {
    event.preventDefault();

    // Add the user's message to the chat history
    setMessages((prevMessages) => [
      ...prevMessages,
      { type: "user", text: prompt },
    ]);

    try {
      // Send the prompt to the Flask server
      const res = await axios.post("http://localhost:5000/api/prompt", {
        prompt: prompt,
      });

      // Add the bot's response to the chat history
      setMessages((prevMessages) => [
        ...prevMessages,
        { type: "bot", text: res.data.response },
      ]);
    } catch (error) {
      console.error("Error sending prompt:", error);

      // Add an error message to the chat history
      setMessages((prevMessages) => [
        ...prevMessages,
        {
          type: "bot",
          text: "Error: Unable to fetch response from the server.",
        },
      ]);
    }

    // Clear the input box
    setPrompt("");
  };

  return (
    <div className="flex min-h-[90vh]">
      {/* Left Content (Chat Container) */}
      <div className="w-3/5 p-4 bg-white border border-gray-300 rounded-lg shadow-md flex flex-col">
        {/* Chat Content */}
        <div className="flex flex-col overflow-y-auto flex-1">
          {messages.map((message, index) => (
            <div
              key={index}
              className={`flex items-start gap-3 p-2 w-[80%] ${
                message.type === "user" ? "self-end" : ""
              }`}
            >
              {/* User Message */}
              {message.type === "user" ? (
                <div className="bg-gray-100 p-4 mb-4 w-full max-w-none rounded text-base font-inter">
                  {message.text}
                </div>
              ) : (
                // Bot Message
                <>
                  <div className="w-12 h-12 rounded-full bg-gray-300 flex items-center justify-center">
                    <img
                      src={Gangastondog}
                      alt="Bot Icon"
                      className="w-12 h-12 rounded-full"
                    />
                  </div>
                  <div className="bg-blue-500 p-4 rounded-lg border border-cyan-400 max-w-lg">
                    <p className="text-white text-base font-inter">
                      {message.text}
                    </p>
                  </div>
                </>
              )}
            </div>
          ))}
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
