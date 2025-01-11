import React from 'react';
import { Filter } from 'lucide-react';
import { mockComponents } from '../data/mockComponents';

interface FiltersProps {
  onFilterChange: (filters: { category?: string; manufacturer?: string }) => void;
}

export const Filters: React.FC<FiltersProps> = ({ onFilterChange }) => {
  const categories = Array.from(new Set(mockComponents.map(c => c.category)));
  const manufacturers = Array.from(new Set(mockComponents.map(c => c.manufacturer)));

  return (
    <div className="flex items-center gap-4 text-slate-100">
      <Filter size={20} className="text-slate-400" />
      <select
        onChange={(e) => onFilterChange({ category: e.target.value })}
        className="bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 focus:outline-none 
                 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
      >
        <option value="">All Categories</option>
        {categories.map(category => (
          <option key={category} value={category}>{category}</option>
        ))}
      </select>

      <select
        onChange={(e) => onFilterChange({ manufacturer: e.target.value })}
        className="bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 focus:outline-none 
                 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
      >
        <option value="">All Manufacturers</option>
        {manufacturers.map(manufacturer => (
          <option key={manufacturer} value={manufacturer}>{manufacturer}</option>
        ))}
      </select>
    </div>
  );
};