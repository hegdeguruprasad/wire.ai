import React from 'react';
import { PDFEmbed } from './PDFEmbed';
import { PDFPlaceholder } from './PDFPlaceholder';

interface PDFViewerProps {
  url?: string;
}

export const PDFViewer: React.FC<PDFViewerProps> = ({ url }) => {
  if (!url) {
    return <PDFPlaceholder />;
  }

  return <PDFEmbed url={url} />;
};