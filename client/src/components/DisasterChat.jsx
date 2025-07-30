import { useState } from "react";
import axios from "axios";
import ArrowOutwardIcon from '@mui/icons-material/ArrowOutward';

const DisasterChat = ({setDisasterData}) => {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([
    { role: "ai", content: "Hello! How can I assist you today?" }
  ]);

  const sendMessage = async () => {
    if (!input.trim()) return;
    setInput("");
    // Add user message to chat
    const newMessages = [...messages, { role: "user", content: input }];
    setMessages(newMessages);

    try {
      const response = await axios.post("http://localhost:8000/ask", {
        query: input
      });

      const aiReply = response.data.response || "No response from AI.";
      setMessages([...newMessages, { role: "ai", content: aiReply }]);
      if (response.data.locations) {
        setDisasterData(response.data.locations);
      }
    } catch (error) {
      console.error("Error:", error);
      setMessages([
        ...newMessages,
        { role: "ai", content: "Error reaching the AI server." }
      ]);
    }
    
    
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter") sendMessage();
  };

  return (
    <div className="w-full md:w-1/4 h-[95vh] p-4 overflow-y-auto bg-white shadow-md flex flex-col">
      <h2 className="text-2xl font-bold mb-4 text-blue-700">AI Disaster Assistant</h2>

      <div className="flex-grow space-y-4 overflow-y-auto pr-2">
        {messages.map((msg, index) => (
          <div key={index} className={msg.role === "user" ? "flex justify-end" : "flex items-start"}>
            <div
              className={`relative text-sm px-4 py-2 rounded-lg max-w-xs ${
                msg.role === "user"
                  ? "bg-blue-600 text-white"
                  : "bg-blue-100 text-gray-800"
              }`}
            >
              {msg.content}
              <div
                className={`absolute top-2 w-0 h-0 border-t-8 border-t-transparent border-b-8 border-b-transparent ${
                  msg.role === "user"
                    ? "right-[-8px] border-l-8 border-l-blue-600"
                    : "left-[-8px] border-r-8 border-r-blue-100"
                }`}
              ></div>
            </div>
          </div>
        ))}
      </div>

      <div className="flex mt-4">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type a message..."
          className="flex-grow border border-gray-300 px-4 py-2 rounded-l-md outline-none focus:border-blue-500"
        />
        <button
          onClick={sendMessage}
          className="bg-blue-600 text-white px-4 py-2 rounded-r-md transition-all duration-300 hover:bg-blue-700 hover:scale-105 active:scale-95"
        >
          <ArrowOutwardIcon />
        </button>
      </div>
    </div>
  );
};

export default DisasterChat;
