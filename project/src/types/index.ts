export interface Component {
  id: string;
  partNumber: string;
  manufacturer: string;
  category: string;
  description: string;
  datasheetUrl: string;
  specifications: Record<string, string>;
}

export interface SearchFilters {
  category?: string;
  manufacturer?: string;
}

export interface ChatMessage {
  id: string;
  content: string;
  sender: 'user' | 'bot';
  timestamp: Date;
}