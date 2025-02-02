
import React, { useState, useEffect, useRef } from 'react';
import { Send } from 'lucide-react';
import { ChatMessage } from '../types';




interface ChatWindowProps {
  messages: ChatMessage[];
  onSendMessage: (message: string) => void;
  isLoading?: boolean;
}





export const ChatWindow: React.FC<ChatWindowProps> = ({ 
  messages, 
  onSendMessage,
  isLoading = false
}) => {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim() && !isLoading) {
      onSendMessage(input);
      setInput('');
    }
  };

  return (
    <div className="flex flex-col h-full bg-black opacity-90 rounded-lg">
      <div className="p-4 border-b border-slate-800">
        <h2 className="text-lg font-semibold text-slate-100">Component Assistant</h2>
      </div>
      
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[80%] p-3 rounded-lg ${
                message.sender === 'user'
                  ? 'bg-[#00ff7f]/10 text-white'
                  : 'bg-slate-800 text-slate-100'
              }`}
            >
              {message.content}
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="flex justify-start">
            <div className="max-w-[80%] p-3 rounded-lg bg-slate-800">
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-[#00ff7f]/60 rounded-full animate-pulse"></div>
                <div className="w-2 h-2 bg-[#00ff7f]/60 rounded-full animate-pulse delay-75"></div>
                <div className="w-2 h-2 bg-[#00ff7f]/60 rounded-full animate-pulse delay-150"></div>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <form onSubmit={handleSubmit} className="p-4 border-t border-slate-800">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={isLoading ? "Processing..." : "Ask about the component..."}
            className="flex-1 px-4 py-2 bg-black border border-slate-700 rounded-lg
                     focus:outline-none focus:ring-2 focus:ring-[#00ff7f]/10 focus:border-transparent
                     text-slate-100 placeholder-slate-400"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={isLoading}
            className={`p-2 bg-gradient-to-b from-[#00ff7f]/10 to-black rounded-lg 
                     transition-colors ${isLoading ? 'opacity-50 cursor-not-allowed' : 'hover:bg-[#00ff7f]/20'}`}
          >
            <Send size={20} className="text-white" />
          </button>
        </div>
      </form>
    </div>
  );
};


