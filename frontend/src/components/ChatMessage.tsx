'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { UserIcon, CpuChipIcon, ExclamationTriangleIcon } from '@heroicons/react/24/outline';

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
    [key: string]: any;
  };
  processingTime?: number;
  confidence?: number;
  sourcesUsed?: {
    document?: number;
    web?: number;
    google_drive?: number;
    [key: string]: number | undefined;
  } | number;
  isError?: boolean;
}

interface ChatMessageProps {
  message: Message;
  onCitationClick: (citationId: string) => void;
}

export default function ChatMessage({ message, onCitationClick }: ChatMessageProps) {
  const isUser = message.role === 'user';
  const isError = message.isError;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}
    >
      <div className={`flex max-w-[80%] ${isUser ? 'flex-row-reverse' : 'flex-row'} items-start space-x-3`}>
        {/* Avatar */}
        <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
          isUser 
            ? 'bg-blue-600' 
            : isError 
              ? 'bg-red-500' 
              : 'bg-gradient-to-r from-purple-500 to-blue-600'
        }`}>
          {isUser ? (
            <UserIcon className="w-5 h-5 text-white" />
          ) : isError ? (
            <ExclamationTriangleIcon className="w-5 h-5 text-white" />
          ) : (
            <CpuChipIcon className="w-5 h-5 text-white" />
          )}
        </div>

        {/* Message Content */}
        <div className={`flex flex-col space-y-2 ${isUser ? 'items-end' : 'items-start'}`}>
          {/* Message Bubble */}
          <div className={`relative px-4 py-3 rounded-2xl ${
            isUser
              ? 'bg-blue-600 text-white'
              : isError
                ? 'bg-red-50 dark:bg-red-900/20 text-red-800 dark:text-red-200 border border-red-200 dark:border-red-800'
                : 'bg-white dark:bg-gray-700 text-gray-900 dark:text-white border border-gray-200 dark:border-gray-600'
          }`}>
            <div className="whitespace-pre-wrap">{message.content}</div>
            
            {/* Citations */}
            {message.citations && message.citations.length > 0 && (
              <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-600">
                <div className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Sources:
                </div>
                <div className="flex flex-wrap gap-2">
                  {message.citations.map((citation) => (
                    <button
                      key={citation.id}
                      onClick={() => onCitationClick(citation.id)}
                      className="text-xs px-2 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-300 rounded-full hover:bg-blue-200 dark:hover:bg-blue-900/50 transition-colors"
                    >
                      {citation.source}
                      {citation.page && ` (p.${citation.page})`}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Metadata */}
          {!isUser && !isError && (message.metadata || message.processingTime || message.confidence !== undefined) && (
            <div className="text-xs text-gray-500 dark:text-gray-400 space-y-1">
              {message.metadata && (
                <div className="flex space-x-4">
                  {message.metadata.sourcesSearched !== undefined && (
                    <span>🔍 {message.metadata.sourcesSearched} sources searched</span>
                  )}
                  {message.metadata.documentsFound !== undefined && (
                    <span>📄 {message.metadata.documentsFound} docs found</span>
                  )}
                  {message.metadata.webResults !== undefined && (
                    <span>🌐 {message.metadata.webResults} web results</span>
                  )}
                </div>
              )}
              <div className="flex space-x-4">
                {message.processingTime && (
                  <span>⏱️ {message.processingTime}ms</span>
                )}
                {message.confidence !== undefined && (
                  <span>🎯 {(message.confidence * 100).toFixed(0)}% confidence</span>
                )}
                {message.sourcesUsed !== undefined && (
                  <span>📚 {
                    typeof message.sourcesUsed === 'number' 
                      ? `${message.sourcesUsed} sources used`
                      : (() => {
                          const sources = message.sourcesUsed as any;
                          const parts = [];
                          if (sources.document > 0) parts.push(`${sources.document} docs`);
                          if (sources.web > 0) parts.push(`${sources.web} web`);
                          if (sources.google_drive > 0) parts.push(`${sources.google_drive} drive`);
                          return parts.length > 0 ? parts.join(', ') : '0 sources used';
                        })()
                  }</span>
                )}
              </div>
            </div>
          )}

          {/* Timestamp */}
          <div className="text-xs text-gray-400 dark:text-gray-500">
            {message.timestamp.toLocaleTimeString([], { 
              hour: '2-digit', 
              minute: '2-digit' 
            })}
          </div>
        </div>
      </div>
    </motion.div>
  );
} 