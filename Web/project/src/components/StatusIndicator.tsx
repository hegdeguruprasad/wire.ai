import React from 'react';
import { Loader2, CheckCircle2, AlertCircle } from 'lucide-react';

interface StatusIndicatorProps {
  status: 'idle' | 'processing' | 'completed' | 'failed';
  componentName?: string;
}

const StatusIndicator: React.FC<StatusIndicatorProps> = ({ status, componentName }) => {
  const getStatusContent = () => {
    switch (status) {
      case 'processing':
        return (
          <div className="flex items-center gap-2 text-[#00ff7f]/80 bg-[#00ff7f]/5 px-4 py-2 rounded-lg">
            <Loader2 className="h-4 w-4 animate-spin" />
            <span>Processing datasheet for {componentName}...</span>
          </div>
        );
      case 'completed':
        return (
          <div className="flex items-center gap-2 text-[#00ff7f] bg-[#00ff7f]/5 px-4 py-2 rounded-lg">
            <CheckCircle2 className="h-4 w-4" />
            <span>Datasheet processed. You can now ask questions!</span>
          </div>
        );
      case 'failed':
        return (
          <div className="flex items-center gap-2 text-red-500 bg-red-500/5 px-4 py-2 rounded-lg">
            <AlertCircle className="h-4 w-4" />
            <span>Failed to process datasheet. Please try again.</span>
          </div>
        );
      default:
        return null;
    }
  };

  return (
    <div className="w-full flex justify-center">
      {getStatusContent()}
    </div>
  );
};

export default StatusIndicator;