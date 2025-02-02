// import React from 'react';
// import { Search } from 'lucide-react';

// interface SearchBarProps {
//   onSearch: (query: string) => void;
// }

// export const SearchBar: React.FC<SearchBarProps> = ({ onSearch }) => {
//   const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
//     e.preventDefault();
//     const formData = new FormData(e.currentTarget);
//     const query = formData.get('search') as string;
//     onSearch(query);
//   };

//   return (
//     <form onSubmit={handleSubmit} className="w-full max-w-2xl">
//       <div className="relative">
//         <input
//           type="text"
//           name="search"
//           placeholder="Search for components (e.g., ATmega328P)"
//           className="w-full px-4 py-3 pl-12 bg-slate-800 border border-slate-700 rounded-lg 
//                    focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent
//                    text-slate-100 placeholder-slate-400"
//         />
//         <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-slate-400" size={20} />
//       </div>
//     </form>
//   );
// };


// SearchBar.tsx
import React, { useState } from 'react';
import { Search } from 'lucide-react';
import { Component } from '../types';

interface SearchBarProps {
  onSearch: (component: Component | null) => void;
}

export const SearchBar: React.FC<SearchBarProps> = ({ onSearch }) => {
  const [isLoading, setIsLoading] = useState(false);

  const searchComponent = async (query: string) => {
    try {
      setIsLoading(true);
      
      const response = await fetch(
         // Uncomment this to run locally
        // `http://localhost:8000/components/search/?query=${encodeURIComponent(query)}`
        `https://wire-ai.fly.dev/components/search/?query=${encodeURIComponent(query)}`
      );
      
      if (response.status === 404) {
        onSearch(null); // Component not found
        return;
      }
      
      if (!response.ok) {
        throw new Error('Search failed');
      }
      
      const component = await response.json();
      onSearch(component);
      
    } catch (error) {
      console.error('Search error:', error);
      onSearch(null);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const formData = new FormData(e.currentTarget);
    const query = formData.get('search') as string;
    if (query.trim()) {
      searchComponent(query);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="w-full max-w-2xl">
      <div className="relative">
        <input
          type="text"
          name="search"
          placeholder="Search by Manufacturer Part Number (e.g., P440-825FM)"
          className="w-full px-4 py-3 pl-12 bg-black border border-slate-700 rounded-lg 
                   focus:outline-none focus:ring-2 focus:ring-[#00ff7f]/10 focus:border-transparent
                   text-slate-100 placeholder-slate-400"
          disabled={isLoading}
        />
        <Search 
          className={`absolute left-4 top-1/2 transform -translate-y-1/2 ${
            isLoading ? 'text-slate-500' : 'text-slate-400'
          }`} 
          size={20} 
        />
      </div>
    </form>
  );
};