import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import Gangastondog from "../assets/images/gangastondog.jpg";

function Home() {
  const [prompt, setPrompt] = useState("");
  const [messages, setMessages] = useState([]);
  const [formData, setFormData] = useState(null);
  const [sessionId, setSessionId] = useState(null);
  const [activeField, setActiveField] = useState(null); // Start with no active field
  const chatContainerRef = useRef(null);

  // Field commands
  const fieldCommands = {
    title: "Extract the title from this text: ",
    firstname: "Extract the firstname from this text: ",
    lastname: "Extract the lastname from this text: ",
    age: "Extract the age from this text: ",
    sex: "Extract the sex from this text: ",
    genderreassignment: "Extract the gender reassignment from this text: ",
    marriagecivilpartnership: "Extract the marriage/civil partnership status from this text: ",
    sexualorientation: "Extract the sexual orientation from this text: ",
    disability: "Extract the disability from this text: ",
    religionbelief: "Extract the religion/belief from this text: ",
    ethnicity: "Extract the ethnicity from this text: ",
    race: "Extract the race from this text: ",
    pregnancymaternity: "Extract the pregnancy/maternity from this text: ",
  };

  // 1. Create session & fetch initial form data on mount
  useEffect(() => {
    const createSessionAndFetchForm = async () => {
      try {
        // Step A: Create a new session
        const sessionRes = await axios.post("http://localhost:5000/api/session", {
          formType: "gp-registration",
        });
        const newSessionId = sessionRes.data.session_id;
        setSessionId(newSessionId);

        // Step B: Fetch the initial form data
        const formRes = await axios.post("http://localhost:5000/api/setup", {
          formType: "gp-registration",
        });
        // If /api/setup returns raw array for "CHARACTERISTICS_ONLY," 
        // or an object for "FULL_FORM," handle accordingly:
        // Let's assume "formRes.data" is an array or an object with "Characteristics".
        // If you are returning { session_id, form }, then adjust accordingly.
        const setupData = formRes.data; 

        // We'll handle the case if it's an array or an object:
        // If "setupData" is an array of characteristics, 
        // we can wrap it in an object for consistency:
        let finalFormData = {};
        if (Array.isArray(setupData)) {
          // It's probably just an array of "Characteristics"
          finalFormData = { Characteristics: setupData };
        } else if (setupData?.Characteristics) {
          // It's a "FULL_FORM" style object
          finalFormData = setupData;
        }

        setFormData(finalFormData);

        // Step C: If there's at least one field, set it as the active field
        const characteristics = finalFormData.Characteristics || [];
        if (characteristics.length > 0) {
          const firstField = characteristics[0].Name.toLowerCase();
          setActiveField(firstField);

          // Also add a message letting the user know
          setMessages([
            {
              type: "bot",
              text: "Hello! I'm DoctorBot. I'm here to help you fill out a GP Registration form for Public Health Scotland. Let's begin firstly by giving us your " + firstField,
            }
          ]);
        } else {
          // If no fields, just keep the initial bot message
          setMessages([
            {
              type: "bot",
              text: "Hello! I'm DoctorBot. I'm here to help you fill out a GP Registration form for Public Health Scotland. Let's begin!",
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

  // 2. Auto-scroll chat messages
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages]);

  // 3. Handle field click
  const handleFieldClick = (field) => {
    setActiveField(field);
    setMessages((prev) => [
      ...prev,
      { type: "bot", text: `You are now editing the ${field} field.` },
    ]);
  };

  // 4. Submit user prompt
  const handleSubmit = async (event) => {
    event.preventDefault();
  
    // Prepend the command for the active field
    // const fieldPrompt = activeField ? fieldCommands[activeField] || "" : "";
    // const fullPrompt = `${fieldPrompt}${prompt}`;
  
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
          return prevValue !== newValue; // Return true if the value has changed
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
            setActiveField(nextField);
  
            // Notify the user about the next field
            setMessages((prev) => [
              ...prev,
              {
                type: "bot",
                text: `Great! Now let's move on to the ${nextField} field.`,
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
                onClick={() => handleFieldClick(fieldName)}
              >
                <td className="border px-3 py-2 font-semibold">{item.Name || "N/A"}</td>
                <td className="border px-3 py-2">
                  {item.Value && String(item.Value).trim() !== "" ? item.Value : "N/A"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    );
  };

  // Main chat UI
  return (
    <div className="flex min-h-[70vh]">
      {/* Left side (Chat Container) */}
      <div className="w-3/5 p-4 bg-white border border-gray-300 rounded-lg shadow-md flex flex-col">
        <div ref={chatContainerRef} className="flex flex-col overflow-y-auto h-[80vh] p-2">
          {messages.map((message, index) => (
            <div
              key={index}
              className={`flex items-start gap-3 p-2 w-[80%] ${
                message.type === "user" ? "self-end" : ""
              }`}
            >
              {message.type === "user" ? (
                <div className="bg-gray-100 p-4 mb-4 w-full max-w-none rounded text-base font-inter">
                  {message.text}
                </div>
              ) : (
                <>
                  <div className="w-12 h-12 rounded-full bg-gray-300 flex items-center justify-center">
                    <img src={Gangastondog} alt="Bot Icon" className="w-12 h-12 rounded-full" />
                  </div>
                  <div className="bg-blue-500 p-4 rounded-lg border border-cyan-400 max-w-lg">
                    <p className="text-white text-base font-inter">{message.text}</p>
                  </div>
                </>
              )}
            </div>
          ))}
        </div>

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
      </div>
    </div>
  );
}

export default Home;
