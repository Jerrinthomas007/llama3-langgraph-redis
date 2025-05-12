import { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { v4 as uuidv4 } from 'uuid';

export default function Chat() {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([]);
  const [userId, setUserId] = useState('');
  const [isWaiting, setIsWaiting] = useState(false);
  const chatEndRef = useRef(null);

  useEffect(() => {
    let storedId = localStorage.getItem('chat_user_id');
    if (!storedId) {
      storedId = uuidv4();
      localStorage.setItem('chat_user_id', storedId);
    }
    setUserId(storedId);
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || isWaiting) return;

    const userMessage = { role: 'user', content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsWaiting(true);

    try {
      const res = await axios.post('http://127.0.0.1:8000/chat', {
        user_id: userId,
        message: input,
      });

      const botMessage = { role: 'bot', content: res.data.response };
      setMessages((prev) => [...prev, botMessage]);
    } catch (err) {
      console.error('API error:', err);
      setMessages((prev) => [
        ...prev,
        { role: 'bot', content: 'Something went wrong. Please try again.' },
      ]);
    } finally {
      setIsWaiting(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') handleSend();
  };

  return (
    <div className="min-h-screen bg-gradient-to-r from-indigo-100 via-purple-100 to-pink-100 font-sans">
      <div className="max-w-2xl mx-auto mt-10 bg-white shadow-2xl rounded-xl overflow-hidden flex flex-col h-[600px]">
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {messages.map((msg, idx) => (
            <div
              key={idx}
              className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`rounded-2xl px-5 py-3 max-w-[75%] whitespace-pre-wrap text-sm shadow ${
                  msg.role === 'user'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-100 text-gray-800'
                }`}
              >
                {msg.content}
              </div>
            </div>
          ))}
          {isWaiting && (
            <div className="text-gray-400 italic text-sm">Bot is typing...</div>
          )}
          <div ref={chatEndRef} />
        </div>

        <div className="border-t bg-white p-4 flex items-center gap-2">
          <input
            type="text"
            className="flex-1 border border-gray-300 rounded-full px-4 py-2 focus:outline-none focus:ring-2 focus:ring-purple-400"
            placeholder="Type a message..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            disabled={isWaiting}
          />
          <button
            onClick={handleSend}
            disabled={isWaiting}
            className={`px-4 py-2 rounded-full font-semibold text-white transition ${
              isWaiting
                ? 'bg-gray-400 cursor-not-allowed'
                : 'bg-purple-600 hover:bg-purple-700'
            }`}
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}
