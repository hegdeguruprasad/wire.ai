
import React, { useState, useEffect, useRef } from 'react';
import { SearchBar } from '../components/SearchBar';
import { ChatWindow } from '../components/ChatWindow';
import { PDFViewer } from '../components/PDFViewer';
import { CircuitBoard } from 'lucide-react';
import { ChatMessage, Component } from '../types';
import { ParticleBackground } from '../components/ParticleBackground';
import StatusIndicator from '../components/StatusIndicator';


function RAGPage() {
  const [selectedComponent, setSelectedComponent] = useState<Component | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [processingStatus, setProcessingStatus] = useState<'idle' | 'processing' | 'completed' | 'failed'>('idle');
  const completionMessageShown = useRef(false);


  

  const checkStatus = async () => {
    if (!selectedComponent?.id) return false;

    try {
      const response = await fetch(
        // Uncomment this to run locally
        // `http://localhost:8000/components/${selectedComponent.id}/datasheet/status`
        `https://wire-ai.fly.dev/components/${selectedComponent.id}/datasheet/status`
      );
      if (response.ok) {
        const status = await response.json();
        if (status.status === 'completed'&& !completionMessageShown.current) {
          setProcessingStatus('completed');
          setSelectedComponent(prev => ({
            ...prev!,
            doc_id: status.doc_id
          }));
          setMessages(prev => [...prev, {
            id: Date.now().toString(),
            content: "Datasheet is now ready. You can ask questions about it.",
            sender: 'bot',
            timestamp: new Date()
          }]);
          completionMessageShown.current = true;
          return true;
        } else if (status.status === 'failed'&& !completionMessageShown.current) {
          setProcessingStatus('failed');
          setMessages(prev => [...prev, {
            id: Date.now().toString(),
            content: "Sorry, there was an error processing the datasheet.",
            sender: 'bot',
            timestamp: new Date()
          }]);
          completionMessageShown.current = true;
          return true;
        }
      }
      return false;
    } catch (error) {
      console.error('Status check error:', error);
      setProcessingStatus('failed');
      return false;
    }
  };

  useEffect(() => {
    let intervalId: number;
    
    if (processingStatus === 'processing' && selectedComponent?.id) {
      completionMessageShown.current = false;
      intervalId = window.setInterval(async () => {
        const done = await checkStatus();
        if (done) {
          clearInterval(intervalId);
        }
      }, 5000);
    }

    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [selectedComponent?.id, processingStatus]);

  const handleSearch = (component: Component | null) => {
    setSelectedComponent(component);
    setMessages([]);
    setProcessingStatus('idle');
    completionMessageShown.current = false;
    
    if (component) {
      console.log('Component found:', component);
      
      let message = `Found component: ${component.ManufacturerPartNumber}.`;
      
      if (component.doc_id) {
        message += " Datasheet is ready. You can ask questions about it.";
        setProcessingStatus('completed');
        completionMessageShown.current = true;
        setMessages([{
          id: Date.now().toString(),
          content: message,
          sender: 'bot',
          timestamp: new Date()
        }]);
      } else if (component.DataSheetUrl) {
        setProcessingStatus('processing');
        message += " Processing datasheet, please wait a moment...";
        setMessages([{
          id: Date.now().toString(),
          content: message,
          sender: 'bot',
          timestamp: new Date()
        }]);
      }
    } else {
      setMessages([{
        id: Date.now().toString(),
        content: 'No component found matching the part number. Please try another search.',
        sender: 'bot',
        timestamp: new Date()
      }]);
    }
  };

  const handleSendMessage = async (message: string) => {
    if (!message.trim()) return;
  
    if (!selectedComponent?.doc_id) {
      setMessages(prev => [...prev, {
        id: Date.now().toString(),
        content: message,
        sender: 'user',
        timestamp: new Date()
      }, {
        id: Date.now().toString() + '1',
        content: "Please wait for the datasheet to finish processing before asking questions.",
        sender: 'bot',
        timestamp: new Date()
      }]);
      return;
    }
  
    setMessages(prev => [...prev, {
      id: Date.now().toString(),
      content: message,
      sender: 'user',
      timestamp: new Date()
    }]);
  
    setIsLoading(true);
  
    try {
      console.log('Sending chat request with doc_id:', selectedComponent.doc_id);
      const response = await fetch(
         // Uncomment this to run locally
        // `http://localhost:8000/documents/${selectedComponent.doc_id}/chat`,
        `https://wire-ai.fly.dev/documents/${selectedComponent.doc_id}/chat`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Connection': 'keep-alive',
          },
          body: JSON.stringify({
            question: message,
            conversation_id: conversationId,
            previous_messages: messages.slice(-6).map(msg => ({
              role: msg.sender === 'user' ? 'user' : 'assistant',
              content: msg.content
            }))
          }),
        }
      );
  
      if (!response.ok) {
        throw new Error(`Chat request failed: ${response.status}`);
      }
  
      const data = await response.json();
      
      if (data.conversation_id && !conversationId) {
        setConversationId(data.conversation_id);
      }
  
      setMessages(prev => [...prev, {
        id: Date.now().toString(),
        content: data.answer,
        sender: 'bot',
        timestamp: new Date()
      }]);
  
    } catch (error) {
      console.error('Chat error:', error);
      setMessages(prev => [...prev, {
        id: Date.now().toString(),
        content: "Sorry, I encountered an error. Please try again.",
        sender: 'bot',
        timestamp: new Date()
      }]);
    } finally {
      setIsLoading(false);
    }
  };






  return (
    <div className="min-h-screen bg-gradient-to-b from-[#00ff7f]/10 to-black text-white">
      <ParticleBackground />


      {/* Logo
      <div className="fixed top-0 left-0 z-50 p-6">
        <Logo />
      </div> */}

      

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8 relative">
        {/* Search Section */}
        <div className="mb-8 flex flex-col items-center gap-6">
          <SearchBar onSearch={handleSearch} />
          {selectedComponent && (
            <>
            <div className="text-sm text-[#00ff7f]/80">
              <span className="font-medium">
                {selectedComponent.ManufacturerPartNumber}
              </span>
              {' • '}
              <span className="text-[#00ff7f]/60">{selectedComponent.Category}</span>
              {' • '}
              <span className="text-[#00ff7f]/60">{selectedComponent.Manufacturer}</span>
            </div>
            <StatusIndicator status={processingStatus} componentName={selectedComponent.ManufacturerPartNumber}/>
            </>
          
          )}
        </div>

        {/* Two-Pane Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 h-[calc(100vh-300px)]">
          {/* Chat Window */}
          <div className="rounded-lg overflow-hidden border border-[#00ff7f]/20 shadow-[0_0_30px_rgba(0,255,127,0.1)]">
            <ChatWindow messages={messages} onSendMessage={handleSendMessage} isLoading={isLoading}/>
          </div>
          
          {/* PDF Viewer */}
          <div className="rounded-lg overflow-hidden border border-[#00ff7f]/20 shadow-[0_0_30px_rgba(0,255,127,0.1)]">
            <PDFViewer url={selectedComponent?.DataSheetUrl} />
          </div>
        </div>
      </main>
    </div>


  );
}

export default RAGPage;