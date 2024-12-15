import React, { useState } from 'react';
import { SearchBar } from './components/SearchBar';
import { ChatWindow } from './components/ChatWindow';
import { PDFViewer } from './components/PDFViewer';
import { Filters } from './components/Filters';
import { mockComponents } from './data/mockComponents';
import { ChatMessage, Component, SearchFilters } from './types';
import { CircuitBoard } from 'lucide-react';

function App() {
  const [selectedComponent, setSelectedComponent] = useState<Component | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [filters, setFilters] = useState<SearchFilters>({});

  const handleSearch = (query: string) => {
    const component = mockComponents.find(c => 
      c.partNumber.toLowerCase().includes(query.toLowerCase()) &&
      (!filters.category || c.category === filters.category) &&
      (!filters.manufacturer || c.manufacturer === filters.manufacturer)
    );
    
    if (component) {
      setSelectedComponent(component);
      setMessages([{
        id: Date.now().toString(),
        content: `I found the ${component.partNumber}. What would you like to know about it?`,
        sender: 'bot',
        timestamp: new Date()
      }]);
    } else {
      setMessages([{
        id: Date.now().toString(),
        content: 'No components found matching your search criteria.',
        sender: 'bot',
        timestamp: new Date()
      }]);
    }
  };

  const handleSendMessage = (message: string) => {
    const newMessages: ChatMessage[] = [
      ...messages,
      {
        id: Date.now().toString(),
        content: message,
        sender: 'user',
        timestamp: new Date()
      }
    ];

    if (selectedComponent) {
      newMessages.push({
        id: (Date.now() + 1).toString(),
        content: `This is a mock response about ${selectedComponent.partNumber}. In a real implementation, this would be handled by a backend service.`,
        sender: 'bot',
        timestamp: new Date()
      });
    }

    setMessages(newMessages);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      {/* Header */}
      <header className="border-b border-slate-800">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center gap-2">
            <CircuitBoard size={24} className="text-blue-500" />
            <h1 className="text-xl font-bold">Electronic Components Viewer</h1>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        {/* Search Section */}
        <div className="mb-8 flex flex-col items-center gap-6">
          <SearchBar onSearch={handleSearch} />
          <Filters onFilterChange={setFilters} />
        </div>

        {/* Two-Pane Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 h-[calc(100vh-300px)]">
          {/* Chat Window */}
          <ChatWindow messages={messages} onSendMessage={handleSendMessage} />
          
          {/* PDF Viewer */}
          <PDFViewer url={selectedComponent?.datasheetUrl} />
        </div>
      </main>
    </div>
  );
}

export default App;