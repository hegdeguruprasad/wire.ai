export interface Component {
  id: string;
  ManufacturerPartNumber: string;
  Category: string;
  DataSheetUrl: string;
  Description: string;
  ImagePath: string;
  Manufacturer: string;
  MouserPartNumber: string;
  Part_Number: string;
  ProductDetailUrl: string;
  ROHSStatus: string;
  last_updated?: string;  // Optional since we're getting ISO string from backend
  doc_id: string;
  
}

export interface ChatMessage {
  id: string;
  content: string;
  sender: 'user' | 'bot';
  timestamp: Date;
}

