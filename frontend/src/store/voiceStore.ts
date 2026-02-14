import { create } from 'zustand';
import { devtools } from 'zustand/middleware';

interface VoiceState {
  isRecording: boolean;
  isSupported: boolean;
  transcript: string;
  error: string | null;
  setRecording: (isRecording: boolean) => void;
  setSupported: (isSupported: boolean) => void;
  setTranscript: (transcript: string) => void;
  setError: (error: string | null) => void;
  reset: () => void;
}

export const useVoiceStore = create<VoiceState>()(
  devtools(
    (set) => ({
      isRecording: false,
      isSupported: false,
      transcript: '',
      error: null,
      
      setRecording: (isRecording) => 
        set(() => ({ isRecording })),
      
      setSupported: (isSupported) => 
        set(() => ({ isSupported })),
      
      setTranscript: (transcript) => 
        set(() => ({ transcript })),
      
      setError: (error) => 
        set(() => ({ error })),
      
      reset: () => 
        set(() => ({
          isRecording: false,
          transcript: '',
          error: null,
        })),
    }),
    {
      name: 'voice-storage',
    }
  )
); 