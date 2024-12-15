import React from 'react';
import { FileText } from 'lucide-react';

interface PDFViewerProps {
  url?: string;
}

export const PDFViewer: React.FC<PDFViewerProps> = ({ url }) => {
  if (!url) {
    return (
      <div className="h-full flex items-center justify-center bg-slate-900 rounded-lg">
        <div className="text-center text-slate-400">
          <FileText size={48} className="mx-auto mb-4" />
          <p>Select a component to view its datasheet</p>
        </div>
      </div>
    );
  }

  return (
    <iframe
      src={url}
      className="w-full h-full rounded-lg bg-slate-900"
      title="Component Datasheet"
    />
  );
};