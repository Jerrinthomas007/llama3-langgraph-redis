import { useState } from 'react';
import axios from 'axios';

export default function Chat() {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([]);
  const userId = 'user123'; // Replace with dynamic user ID if needed

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage = { role: 'user', content: input };
    setMessages((prev) => [...prev, userMessage]);

    try {
      const res = await axios.post('http://127.0.0.1:8000/chat', {
        user_id: userId,
        message: input,
      });

      const botMessage = {
        role: 'bot',
        content: res.data.response,
      };

      setMessages((prev) => [...prev, botMessage]);
      setInput('');
    } catch (err) {
      console.error('API error:', err);
    }
  };

  return (
    <div className="p-4 max-w-xl mx-auto">
      <div className="h-96 overflow-y-auto border rounded-lg p-4 mb-4 bg-white shadow">
        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`mb-2 text-sm ${
              msg.role === 'user' ? 'text-blue-600' : 'text-green-600'
            }`}
          >
            <strong>{msg.role === 'user' ? 'You' : 'Bot'}:</strong> {msg.content}
          </div>
        ))}
      </div>
      <div className="flex">
        <input
          type="text"
          className="flex-1 border rounded-l-lg p-2"
          placeholder="Type a message..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
        />
        <button
          className="bg-blue-500 text-white px-4 rounded-r-lg"
          onClick={handleSend}
        >
          Send
        </button>
      </div>
    </div>
  );
}
