import { useState, useRef, useEffect } from "react";
import axios from "axios";
import ArrowOutwardIcon from '@mui/icons-material/ArrowOutward';

const DisasterChat = ({ setDisasterData }) => {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([
    { role: "ai", content: "Hello! How can I assist you today?" }
  ]);
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null); // 👈 for auto scroll

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

 const formatMarkdownToHTML = (text) => {
  if (!text || typeof text !== 'string') return '';
  
  // Clean the text
  let formatted = text.trim();
  
  // Remove hashtags at the beginning of lines
  formatted = formatted.replace(/^#[^\s*]+/gm, "");
  
  // Handle **bold** text
  formatted = formatted.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  
  // Handle *italic* text (but not bullet points)
  formatted = formatted.replace(/(?<!\*)\*([^*]+)\*(?!\*)/g, '<em>$1</em>');
  
  // Split into paragraphs and process each
  const paragraphs = formatted.split(/\n\s*\n/);
  
  return paragraphs.map(paragraph => {
    paragraph = paragraph.trim();
    if (!paragraph) return '';
    
    // Check if this paragraph contains bullet points
    if (paragraph.includes('* ')) {
      // Extract bullet points
      const lines = paragraph.split('\n');
      const bulletPoints = [];
      let currentText = '';
      
      lines.forEach(line => {
        line = line.trim();
        if (line.startsWith('* ')) {
          // This is a bullet point
          if (currentText) {
            // Add any preceding text as a paragraph
            bulletPoints.push(`<p class="mb-3">${currentText}</p>`);
            currentText = '';
          }
          bulletPoints.push(`<li>${line.substring(2).trim()}</li>`);
        } else if (line) {
          // This is regular text
          currentText += (currentText ? ' ' : '') + line;
        }
      });
      
      // Add any remaining text
      if (currentText) {
        bulletPoints.unshift(`<p class="mb-3">${currentText}</p>`);
      }
      
      // Check if we have actual list items
      const listItems = bulletPoints.filter(item => item.startsWith('<li>'));
      const textContent = bulletPoints.filter(item => item.startsWith('<p>'));
      
      if (listItems.length > 0) {
        return textContent.join('') + 
               `<ul class="list-disc pl-5 space-y-2 mb-4">${listItems.join('')}</ul>`;
      } else {
        return textContent.join('');
      }
    } else {
      // Regular paragraph - check for special formatting
      if (paragraph.includes('**') || paragraph.includes('*')) {
        // Already processed bold/italic above
        return `<p class="mb-3">${paragraph}</p>`;
      } else {
        return `<p class="mb-3">${paragraph}</p>`;
      }
    }
  }).join('');
};


  const sendMessage = async () => {
    if (!input.trim()) return;
    setInput("");

    const newMessages = [...messages, { role: "user", content: input }];
    setMessages(newMessages);
    setLoading(true); // start loading

    try {
      const response = await axios.post("http://localhost:8000/ask", {
        query: input
      });
      
      const aiReply = response.data.response || "No response from AI.";
      const htmlFormatted = formatMarkdownToHTML(aiReply);
      console.log("AI Response:", aiReply);
      setMessages([
        ...newMessages,
        { role: "ai", content: htmlFormatted, isHTML: true }
      ]);
      setLoading(false);

      if (response.data.locations) {
        setDisasterData(response.data.locations);
      }

    } catch (error) {
      console.error("Error:", error);
      setMessages([
        ...newMessages,
        { role: "ai", content: "Error reaching the AI server." }
      ]);
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter") sendMessage();
  };

  return (
    <div className="w-full md:w-2/4 h-[95vh] p-4 overflow-y-auto bg-white shadow-md flex flex-col">
      <h2 className="text-2xl font-bold mb-4 text-blue-700">AI Disaster Assistant</h2>

      <div className="flex-grow space-y-4 overflow-y-auto pr-2">
        {messages.map((msg, index) => (
          <div key={index} className={msg.role === "user" ? "flex justify-end" : "flex items-start"}>
            <div
              className={`relative text-sm px-4 py-2 rounded-lg w-90 whitespace-pre-wrap ${
                msg.role === "user"
                  ? "bg-blue-600 text-white"
                  : "bg-blue-100 text-gray-800"
              }`}
            >
              {msg.isHTML ? (
                <div dangerouslySetInnerHTML={{ __html: msg.content }} />
              ) : (
                msg.content
              )}
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

        {loading && (
          <div className="flex items-start">
            <div className="relative text-sm px-4 py-2 rounded-lg w-90 bg-blue-100 text-gray-800 animate-pulse">
              AI is thinking...
              <div className="absolute top-2 w-0 h-0 border-t-8 border-t-transparent border-b-8 border-b-transparent left-[-8px] border-r-8 border-r-blue-100"></div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
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
