import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import DoctorIcon from "../assets/images/doctoricon.png";
import SpeechRecognition, { useSpeechRecognition } from 'react-speech-recognition';
import { useNavigate } from 'react-router-dom';

function Home() {
  const [prompt, setPrompt] = useState("");
  const [messages, setMessages] = useState([]);
  const [formData, setFormData] = useState(null);
  const [sessionId, setSessionId] = useState(null);
  const navigate = useNavigate();
  const [activeField, setActiveField] = useState(null); // Start with no active field
  const [suggestions, setSuggestions] = useState([
    "Who can see my personal information?",
    "How does this data help improve healthcare services?",
    "What can you do?"
  ]);
  
  const chatContainerRef = useRef(null);

  // Create session & fetch initial form data on mount
  useEffect(() => {
    const createSessionAndFetchForm = async () => {
      try {
        // Step 1: Create a new session
        const sessionRes = await axios.post("http://localhost:5000/api/setup", {
          formType: "Characteristics",
        });
        
        const { session_id, form_data } = sessionRes.data;
        
        // Store session ID
        setSessionId(session_id);
        
        // Ensure the form data is in the expected format
        let finalFormData = {};
        if (Array.isArray(form_data)) {
          finalFormData = { Characteristics: form_data };
        } else if (form_data?.Characteristics) {
          finalFormData = form_data;
        }

        setFormData(finalFormData);

        // Step 2: If there's at least one field, set it as the active field
        const characteristics = finalFormData.Characteristics || [];
        if (characteristics.length > 0) {
          const firstField = characteristics[0].Name.toLowerCase();
          setActiveField(firstField);

          // Also add a message letting the user know
          setMessages([
            {
              type: "bot",
              text:
                "Hello! I'm here to help you fill out this equalities form. \n \n Your responses about characteristics help us understand your unique needs and offer you the best possible healthcare experience. All your information remains confidential, secure, and will be treated sensitively. You can type 'Prefer Not To Say' for questions you don't want to answer. \n \n If you would like to view our privacy statement please refer to: https://publichealthscotland.scot/our-privacy-notice/personal-data-collection/  \n \n Let's begin firstly by giving us your " +
                firstField +
                ":",
            },
          ]);
        } else {
          setMessages([
            {
              type: "bot",
              text:
                "Hello! I'm DoctorBot. I'm here to help you fill out a GP Registration form for Public Health Scotland. Let's begin!",
            },
          ]);
        }
      } catch (error) {
        console.error("Error initializing session or fetching form data:", error);
        setMessages([
          {
            type: "bot",
            text: "Error: Unable to load session or form data.",
          },
        ]);
      }
    };

    createSessionAndFetchForm();
  }, []);

  // New useEffect to check if all fields are filled and add a bot message if so
  useEffect(() => {
    if (formData && formData.Characteristics && formData.Characteristics.length > 0) {
      const allFilled = formData.Characteristics.every(
        (item) => item.Value && String(item.Value).trim() !== ""
      );
      const thankYouMessage = "Thank you for filling out the form. Please ensure that all fields are correct before submitting.";
      if (allFilled && !messages.some((msg) => msg.type === "bot" && msg.text.includes("Thank you for filling out the form"))) {
        setMessages((prev) => [...prev, { type: "bot", text: thankYouMessage }]);
      }
    }
  }, [formData, messages]);

  // Auto-scroll chat messages
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages]);

  // 3. Handle field click
  const handleFieldClick = (item) => {
    const potentialDescription = item.Description;
    let chatString = `We are now going to ask you about ${item.Label}. `;
    if (potentialDescription){
      chatString += '\n \n ' + item.Label + ' Definition: ' + potentialDescription;
    }
    setActiveField(item.Name.toLowerCase());
    setMessages((prev) => [
      ...prev,
      { type: "bot", text: chatString },
    ]);
  };

  // Handle Submit Form
  const handleSubmitForm = async () => {
    try {
      const res = await axios.post("http://localhost:5000/api/finish", {
        session_id: sessionId,
      });
      console.log(res.data.message);
      
    } catch (error) {
      console.error("Error finishing session:", error);
      // Optionally display an error message to the user here
    }
    navigate('/FormComplete');
  };

  // Submit user prompt
  const handleSubmit = async (event) => {
    event.preventDefault();

    // Add user message to chat
    setMessages((prev) => [...prev, { type: "user", text: prompt }]);

    try {
      // Send prompt & session ID to Flask
      const res = await axios.post("http://localhost:5000/api/prompt", {
        prompt: prompt,
        field: activeField,
        session_id: sessionId,
      });

      const { response, updated_form } = res.data;

      // Add bot response to chat
      setMessages((prev) => [...prev, { type: "bot", text: response }]);

      // If there's an updated form, check for changes and move to the next field
      if (updated_form) {
        const previousForm = formData;
        const updatedCharacteristics = updated_form.Characteristics;

        // Check if any field value was updated
        const fieldUpdated = updatedCharacteristics.some((item, index) => {
          const prevValue = previousForm.Characteristics[index]?.Value || "";
          const newValue = item.Value || "";
          return prevValue !== newValue;
        });

        if (fieldUpdated) {
          setFormData(updated_form);

          // Automatically move to the next field
          const characteristics = updatedCharacteristics;
          const currentIndex = characteristics.findIndex(
            (item) => item.Name.toLowerCase() === activeField
          );
          if (currentIndex !== -1 && currentIndex + 1 < characteristics.length) {
            const nextField = characteristics[currentIndex + 1].Name.toLowerCase();
            const nextFieldLabel = characteristics[currentIndex + 1].Label;
            const potentialDescription = characteristics[currentIndex + 1].Description;
            let chatString = `Thank you, we are now going to ask you about ${nextFieldLabel}.`;
            if (potentialDescription){
              chatString += '\n \n Definition: ' + potentialDescription;
            }
            setActiveField(nextField);

            // Notify the user about the next field
            setMessages((prev) => [
              ...prev,
              {
                type: "bot",
                text: chatString,
              },
            ]);
          }
        }
      }
    } catch (error) {
      console.error("Error sending prompt:", error);

      // Add error message to the chat
      setMessages((prev) => [
        ...prev,
        { type: "bot", text: "Error: Unable to fetch response from the server." },
      ]);
    }

    // Clear input
    setPrompt("");
  };

  // Mic Functionality
  const {
    transcript,
    listening,
    resetTranscript,
    browserSupportsSpeechRecognition
  } = useSpeechRecognition();

  useEffect(() => {
    setPrompt(transcript);
  }, [transcript]);

  if (!browserSupportsSpeechRecognition) {
    return <div>Your browser does not support speech recognition.</div>;
  }
  
  const handleMicToggle = () => {
    if (listening) {
      SpeechRecognition.stopListening();
    } else {
      resetTranscript();
      SpeechRecognition.startListening({ continuous: true });
    }
  };

  // Handle suggestion bubble click
  const handleSuggestionClick = async (suggestion) => {
    setSuggestions((prev) => prev.filter((s) => s !== suggestion));
    setMessages((prev) => [...prev, { type: "user", text: suggestion }]);

    try {
      const res = await axios.post("http://localhost:5000/api/prompt", {
        prompt: suggestion,
        field: activeField,
        session_id: sessionId,
      });

      const { response, updated_form } = res.data;
      setMessages((prev) => [...prev, { type: "bot", text: response }]);

      if (updated_form) {
        const previousForm = formData;
        const updatedCharacteristics = updated_form.Characteristics;

        const fieldUpdated = updatedCharacteristics.some((item, index) => {
          const prevValue = previousForm.Characteristics[index]?.Value || "";
          const newValue = item.Value || "";
          return prevValue !== newValue;
        });

        if (fieldUpdated) {
          setFormData(updated_form);

          const characteristics = updatedCharacteristics;
          const currentIndex = characteristics.findIndex(
            (item) => item.Name.toLowerCase() === activeField
          );
          if (currentIndex !== -1 && currentIndex + 1 < characteristics.length) {
            const nextField = characteristics[currentIndex + 1].Name.toLowerCase();
            const nextFieldLabel = characteristics[currentIndex + 1].Label;
            setActiveField(nextField);

            setMessages((prev) => [
              ...prev,
              {
                type: "bot",
                text: `Great! Now let's move on to the ${nextFieldLabel} field.`,
              },
            ]);
          }
        }
      }
    } catch (error) {
      console.error("Error sending suggestion prompt:", error);
      setMessages((prev) => [
        ...prev,
        { type: "bot", text: "Error: Unable to fetch response from the server." },
      ]);
    }
  };

  // Render the table with form fields
  const renderCharacteristicsTable = () => {
    if (!formData || !formData.Characteristics) {
      return <p>No fields to display. Please wait while the form data loads.</p>;
    }

    const characteristics = formData.Characteristics;
    if (characteristics.length === 0) {
      return <p>No fields to display.</p>;
    }

    return (
      <table className="table-auto w-full border-collapse border border-gray-300 shadow-sm">
        <thead>
          <tr className="bg-gray-200">
            <th className="border px-3 py-2 w-1/2 text-left">Field</th>
            <th className="border px-3 py-2 w-1/2 text-left">Value</th>
          </tr>
        </thead>
        <tbody>
          {characteristics.map((item, index) => {
            const fieldName = item.Name ? item.Name.toLowerCase() : "";
            return (
              <tr
                key={index}
                className={`cursor-pointer ${activeField === fieldName ? "bg-blue-100" : ""}`}
                onClick={() => handleFieldClick(item)}
              >
                <td className="border px-3 py-2 font-semibold">{item.Label || ""}</td>
                <td className="border px-3 py-2">
                  {item.Value && String(item.Value).trim() !== "" ? item.Value : ""}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    );
  };

  // Determine if the form is fully filled
  const isFormComplete = formData && formData.Characteristics && formData.Characteristics.every(
    (item) => item.Value && String(item.Value).trim() !== ""
  );

  // Main chat UI
  return (
    <div className="flex min-h-[70vh]">
      {/* Left side (Chat Container) */}
      <div className="w-3/5 p-4 bg-white border border-gray-300 rounded-lg shadow-md flex flex-col">
        <div ref={chatContainerRef} className="flex flex-col overflow-y-auto h-[70vh] p-2">
          {messages.map((message, index) => (
            <div
              key={index}
              className={`flex items-start gap-3 p-2 w-[80%] ${message.type === "user" ? "self-end" : ""}`}
            >
              {message.type === "user" ? (
                <div className="bg-gray-100 p-4 mb-4 w-full max-w-none rounded text-base font-inter">
                  {message.text}
                </div>
              ) : (
                <>
                  <div className="w-12 h-12 rounded-full bg-gray-300 flex items-center justify-center aspect-square">
                    <img src={DoctorIcon} alt="Bot Icon" className="w-12 h-12 rounded-full" />
                  </div>
                  <div className="bg-blue-500 p-4 rounded-lg border border-cyan-400 max-w-lg">
                    <p style={{ whiteSpace: 'pre-line' }} className="text-white text-base font-inter">{message.text}</p>
                  </div>
                </>
              )}
            </div>
          ))}
        </div>

        {/* Suggestion Bubbles */}
        {suggestions.length > 0 && (
          <div className="flex flex-wrap gap-2 p-2">
            {suggestions.map((suggestion, index) => (
              <div
                key={index}
                onClick={() => handleSuggestionClick(suggestion)}
                className="cursor-pointer bg-blue-500 hover:bg-blue-600 px-3 py-1 rounded"
              >
                <p className="text-sm text-white">{suggestion}</p>
              </div>
            ))}
          </div>
        )}

        {/* Prompt Input Form */}
        <form onSubmit={handleSubmit} className="flex items-center gap-2 mt-auto">
          <textarea
            className="w-full p-2 rounded border border-gray-300 text-base"
            rows="2"
            placeholder="Type your input here..."
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
          ></textarea>

          <button
            type="button"
            onClick={handleMicToggle}
            className="bg-red-500 text-white px-4 py-2 rounded cursor-pointer text-sm hover:bg-red-700"
          >
            {listening ? "Stop" : "Mic"}
          </button>

          <button
            type="submit"
            className="bg-blue-500 text-white px-4 py-2 rounded cursor-pointer text-sm whitespace-nowrap hover:bg-blue-700"
          >
            Send
          </button>
        </form>
      </div>

      {/* Right side (Form Preview) */}
      <div className="w-2/5 bg-gray-100 p-4">
        <h2 className="text-xl font-bold mb-4">Form</h2>
        {renderCharacteristicsTable()}
        {isFormComplete && (
          <div className="mt-4">
            <div className="bg-green-100 text-green-800 p-4 rounded">
              Thank you for filling out the form. Please ensure that all fields are correct before submitting.
            </div>
            <button
              className="bg-green-500 text-white px-4 py-2 mt-2 rounded w-full"
              onClick={handleSubmitForm}
            >
              Submit Form
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export default Home;
