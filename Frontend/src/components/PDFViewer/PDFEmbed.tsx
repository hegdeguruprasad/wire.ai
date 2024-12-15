import React from 'react';

interface PDFEmbedProps {
  url: string;
}

export const PDFEmbed: React.FC<PDFEmbedProps> = ({ url }) => {
  return (
    <iframe
      src={url}
      className="w-full h-full rounded-lg bg-slate-900"
      title="Component Datasheet"
    />
  );
};