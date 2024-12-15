import React from 'react';
import { FileText } from 'lucide-react';

export const PDFPlaceholder: React.FC = () => (
  <div className="h-full flex items-center justify-center bg-slate-900 rounded-lg">
    <div className="text-center text-slate-400">
      <FileText size={48} className="mx-auto mb-4" />
      <p>Select a component to view its datasheet</p>
    </div>
  </div>
);