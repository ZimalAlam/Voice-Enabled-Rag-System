import { create } from 'zustand';
import { devtools } from 'zustand/middleware';

interface Citation {
  id: string;
  source: string;
  content: string;
  page?: number;
  url?: string;
}

interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: Date;
  citations?: Citation[];
  metadata?: {
    sourcesSearched?: number;
    documentsFound?: number;
    webResults?: number;
  };
  processingTime?: number;
  confidence?: number;
  sourcesUsed?: number;
  isError?: boolean;
}

interface ChatState {
  messages: Message[];
  addMessage: (message: Message) => void;
  clearMessages: () => void;
  updateMessage: (id: string, updates: Partial<Message>) => void;
  removeMessage: (id: string) => void;
}

export const useChatStore = create<ChatState>()(
  devtools(
    (set) => ({
      messages: [],
      
      addMessage: (message) => 
        set((state) => ({
          messages: [...state.messages, message],
        })),
      
      clearMessages: () => 
        set(() => ({
          messages: [],
        })),
      
      updateMessage: (id, updates) =>
        set((state) => ({
          messages: state.messages.map((msg) =>
            msg.id === id ? { ...msg, ...updates } : msg
          ),
        })),
      
      removeMessage: (id) =>
        set((state) => ({
          messages: state.messages.filter((msg) => msg.id !== id),
        })),
    }),
    {
      name: 'chat-storage',
    }
  )
); 