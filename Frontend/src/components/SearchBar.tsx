import React from 'react';
import { Search } from 'lucide-react';

interface SearchBarProps {
  onSearch: (query: string) => void;
}

export const SearchBar: React.FC<SearchBarProps> = ({ onSearch }) => {
  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const formData = new FormData(e.currentTarget);
    const query = formData.get('search') as string;
    onSearch(query);
  };

  return (
    <form onSubmit={handleSubmit} className="w-full max-w-2xl">
      <div className="relative">
        <input
          type="text"
          name="search"
          placeholder="Search for components (e.g., ATmega328P)"
          className="w-full px-4 py-3 pl-12 bg-slate-800 border border-slate-700 rounded-lg 
                   focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent
                   text-slate-100 placeholder-slate-400"
        />
        <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-slate-400" size={20} />
      </div>
    </form>
  );
};