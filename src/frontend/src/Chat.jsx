import { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { v4 as uuidv4 } from 'uuid';
import { Bot, User } from 'lucide-react'; // Optional for icons (needs `lucide-react`)

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
    <div className="min-h-screen bg-gradient-to-br from-purple-100 via-white to-blue-100 flex items-center justify-center px-4">
      <div className="w-full max-w-3xl h-[600px] bg-white rounded-3xl shadow-2xl border border-gray-200 flex flex-col overflow-hidden">
        <div className="flex-1 overflow-y-auto px-6 py-6 space-y-4">
          {messages.map((msg, idx) => (
            <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className="flex items-end gap-2">
                {msg.role === 'bot' && (
                  <div className="w-8 h-8 bg-purple-200 rounded-full flex items-center justify-center text-purple-700">
                    <Bot size={16} />
                  </div>
                )}
                <div
                  className={`rounded-2xl px-4 py-3 max-w-[75%] whitespace-pre-wrap text-sm shadow-md ${
                    msg.role === 'user'
                      ? 'bg-purple-500 text-white'
                      : 'bg-gray-100 text-gray-800'
                  }`}
                >
                  {msg.content}
                </div>
                {msg.role === 'user' && (
                  <div className="w-8 h-8 bg-purple-500 rounded-full flex items-center justify-center text-white">
                    <User size={16} />
                  </div>
                )}
              </div>
            </div>
          ))}
          {isWaiting && (
            <div className="text-gray-400 italic text-sm">Bot is typing...</div>
          )}
          <div ref={chatEndRef} />
        </div>

        <div className="border-t bg-white p-4 flex items-center gap-3">
          <input
            type="text"
            className="flex-1 border border-gray-300 rounded-full px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-purple-400"
            placeholder="Type a message..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            disabled={isWaiting}
          />
          <button
            onClick={handleSend}
            disabled={isWaiting}
            className={`px-5 py-2 rounded-full text-sm font-semibold text-white transition ${
              isWaiting
                ? 'bg-gray-400 cursor-not-allowed'
                : 'bg-gradient-to-r from-purple-500 to-purple-700 hover:from-purple-600 hover:to-purple-800'
            }`}
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}
