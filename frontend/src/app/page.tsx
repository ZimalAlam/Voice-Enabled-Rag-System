'use client';

import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  MicrophoneIcon, 
  PaperAirplaneIcon,
  DocumentPlusIcon,
  Cog6ToothIcon,
  ChatBubbleBottomCenterTextIcon
} from '@heroicons/react/24/outline';
import { MicrophoneIcon as MicrophoneSolidIcon } from '@heroicons/react/24/solid';

import ChatMessage from '@/components/ChatMessage';
import VoiceInput from '@/components/VoiceInput';
import DocumentUpload from '@/components/DocumentUpload';
import CitationModal from '@/components/CitationModal';
import LoadingIndicator from '@/components/LoadingIndicator';
import { useChatStore } from '@/store/chatStore';
import { useVoiceStore } from '@/store/voiceStore';
import { sendQuery } from '@/lib/api';

export default function HomePage() {
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [showUpload, setShowUpload] = useState(false);
  const [selectedCitation, setSelectedCitation] = useState<string | null>(null);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  
  // Store hooks
  const { messages, addMessage, clearMessages } = useChatStore();
  const { isRecording, isSupported: voiceSupported } = useVoiceStore();

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (query: string) => {
    if (!query.trim() || isLoading) return;

    // Add user message
    const userMessage = {
      id: Date.now().toString(),
      content: query,
      role: 'user' as const,
      timestamp: new Date(),
    };
    addMessage(userMessage);

    setIsLoading(true);
    setInputText('');

    try {
      const response = await sendQuery(query);
      
      // Add AI response with properly mapped citations
      const aiMessage = {
        id: (Date.now() + 1).toString(),
        content: response.answer,
        role: 'assistant' as const,
        timestamp: new Date(),
        citations: response.citations?.map((citation: any) => ({
          id: citation.id,
          source: citation.title,
          content: citation.content,
          page: citation.page_number,
          url: citation.url,
        })) || [],
        metadata: response.metadata,
        processingTime: response.processing_time_ms,
        confidence: response.confidence_score,
        sourcesUsed: response.sources_used as any,
      };
      addMessage(aiMessage);

    } catch (error) {
      console.error('Query failed:', error);
      
      // Add error message
      const errorMessage = {
        id: (Date.now() + 1).toString(),
        content: 'I encountered an error while processing your request. Please try again.',
        role: 'assistant' as const,
        timestamp: new Date(),
        isError: true,
      };
      addMessage(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFormSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    handleSubmit(inputText);
  };

  const handleVoiceResult = (transcript: string) => {
    if (transcript.trim()) {
      handleSubmit(transcript);
    }
  };

  const handleCitationClick = (citationId: string) => {
    setSelectedCitation(citationId);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(inputText);
    }
  };

  return (
    <div className="flex h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800">
      {/* Sidebar */}
      <div className="w-80 bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 flex flex-col">
        {/* Header */}
        <div className="p-6 border-b border-gray-200 dark:border-gray-700">
          <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-black rounded-full flex items-center justify-center mx-auto mb-4">

              <ChatBubbleBottomCenterTextIcon className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-900 dark:text-white">
                Agentic RAG
              </h1>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                AI Assistant with Voice & Documents
              </p>
            </div>
          </div>
        </div>

        {/* Controls */}
        <div className="p-4 space-y-3">
          <button
            onClick={() => setShowUpload(true)}
            className="w-full flex items-center space-x-3 px-4 py-3 bg-white text-black rounded-lg hover:bg-gray-100 dark:bg-black/20 transition-colors"          >
            <DocumentPlusIcon className="w-5 h-5" />
            <span className="font-medium">Upload Documents</span>
          </button>

          <button
            onClick={clearMessages}
            className="w-full flex items-center space-x-3 px-4 py-3 bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-600 transition-colors"
          >
            <Cog6ToothIcon className="w-5 h-5" />
            <span className="font-medium">New Conversation</span>
          </button>
        </div>

        {/* Stats */}
        <div className="p-4 border-t border-gray-200 dark:border-gray-700 mt-auto">
          <div className="text-sm text-gray-500 dark:text-gray-400">
            <div className="flex justify-between">
              <span>Messages:</span>
              <span>{messages.length}</span>
            </div>
            {voiceSupported && (
              <div className="flex justify-between mt-1">
                <span>Voice Input:</span>
                <span className="text-green-600 dark:text-green-400">Available</span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          <AnimatePresence>
            {messages.length === 0 ? (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="text-center py-12"
              >
              <div className="w-16 h-16 bg-black rounded-full flex items-center justify-center mx-auto mb-4">
                  <ChatBubbleBottomCenterTextIcon className="w-8 h-8 text-white" />
                </div>
                <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
                  Welcome to Agentic RAG
                </h2>
                <p className="text-gray-600 dark:text-gray-400 max-w-md mx-auto">
                  Ask me anything! I can search through your documents, web sources, and Google Drive to provide comprehensive answers with citations.
                </p>
                <div className="mt-6 flex flex-wrap justify-center gap-3">
                  <span className="px-3 py-1 bg-black text-white rounded-full text-sm">
                    Document Search
                  </span>
                  <span className="px-3 py-1 bg-black text-white rounded-full text-sm">
                    Web Search
                  </span>
                  <span className="px-3 py-1 bg-black text-white rounded-full text-sm">
                    Voice Input
                  </span>
                  <span className="px-3 py-1 bg-black text-white rounded-full text-sm">
                    Google Drive
                  </span>
                </div>
              </motion.div>
            ) : (
              messages.map((message) => (
                <ChatMessage
                  key={message.id}
                  message={message}
                  onCitationClick={handleCitationClick}
                />
              ))
            )}
          </AnimatePresence>
          
          {isLoading && <LoadingIndicator />}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="border-t border-gray-200 dark:border-gray-700 p-6">
          <form onSubmit={handleFormSubmit} className="flex items-end space-x-4">
            {/* Voice Input */}
            {voiceSupported && (
              <VoiceInput
                onResult={handleVoiceResult}
                className="flex-shrink-0"
              />
            )}

            {/* Text Input */}
            <div className="flex-1">
              <div className="relative">
                <input
                  ref={inputRef}
                  type="text"
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder={isRecording ? "Listening..." : "Ask me anything..."}
                  disabled={isLoading || isRecording}
                  className="w-full px-4 py-3 pr-12 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 disabled:opacity-50"
                />
                <button
                  type="submit"
                  disabled={!inputText.trim() || isLoading || isRecording}
                  className="absolute right-2 top-1/2 transform -translate-y-1/2 p-2 text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <PaperAirplaneIcon className="w-5 h-5" />
                </button>
              </div>
            </div>
          </form>
          
          {/* Hints */}
          <div className="mt-3 flex flex-wrap gap-2">
            <button
              onClick={() => setInputText("What are the key findings in my uploaded documents?")}
              className="px-3 py-1 text-xs bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-full hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
            >
              Analyze documents
            </button>
            <button
              onClick={() => setInputText("What's the latest news about AI?")}
              className="px-3 py-1 text-xs bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-full hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
            >
              Search web
            </button>
            <button
              onClick={() => setInputText("Summarize my recent presentations")}
              className="px-3 py-1 text-xs bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-full hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
            >
              Review Google Drive
            </button>
          </div>
        </div>
      </div>

      {/* Modals */}
      <DocumentUpload 
        isOpen={showUpload}
        onClose={() => setShowUpload(false)}
        onUploadSuccess={(uploadedFiles) => {
          // Add a success message to the chat
          const successMessage = {
            id: Date.now().toString(),
            role: 'assistant' as const,
            content: `✅ Successfully uploaded ${uploadedFiles.length} document${uploadedFiles.length > 1 ? 's' : ''}:\n${uploadedFiles.map(f => `• ${f.filename}`).join('\n')}\n\nYou can now ask questions about these documents!`,
            timestamp: new Date(),
            sourcesUsed: uploadedFiles.length,
            citations: [],
            processingTime: 0,
            confidence: 1.0
          };
          addMessage(successMessage);
        }}
      />
      
      <CitationModal
        citationId={selectedCitation}
        isOpen={!!selectedCitation}
        onClose={() => setSelectedCitation(null)}
      />
    </div>
  );
} 