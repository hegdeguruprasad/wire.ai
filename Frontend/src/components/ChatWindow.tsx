import React, { useState } from 'react';
import { Send } from 'lucide-react';
import { ChatMessage } from '../types';

interface ChatWindowProps {
  messages: ChatMessage[];
  onSendMessage: (message: string) => void;
}

export const ChatWindow: React.FC<ChatWindowProps> = ({ messages, onSendMessage }) => {
  const [input, setInput] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim()) {
      onSendMessage(input);
      setInput('');
    }
  };

  return (
    <div className="flex flex-col h-full bg-slate-900 rounded-lg">
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
                  ? 'bg-blue-600 text-white'
                  : 'bg-slate-800 text-slate-100'
              }`}
            >
              {message.content}
            </div>
          </div>
        ))}
      </div>

      <form onSubmit={handleSubmit} className="p-4 border-t border-slate-800">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about the component..."
            className="flex-1 px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg
                     focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent
                     text-slate-100 placeholder-slate-400"
          />
          <button
            type="submit"
            className="p-2 bg-blue-600 rounded-lg hover:bg-blue-700 transition-colors"
          >
            <Send size={20} className="text-white" />
          </button>
        </div>
      </form>
    </div>
  );
};