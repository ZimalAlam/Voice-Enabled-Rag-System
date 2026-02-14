'use client';

import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  XMarkIcon,
  DocumentTextIcon,
  LinkIcon,
  ClipboardDocumentIcon
} from '@heroicons/react/24/outline';

interface CitationModalProps {
  citationId: string | null;
  isOpen: boolean;
  onClose: () => void;
}

// Mock citation data - in a real app, this would come from a store or API
const getCitationData = (id: string | null) => {
  if (!id) return null;
  
  // Mock data
  const citations: Record<string, any> = {
    '1': {
      id: '1',
      title: 'Introduction to Machine Learning',
      source: 'ML_Textbook.pdf',
      content: 'Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from and make predictions on data. It involves training models on historical data to identify patterns and make informed decisions about new, unseen data.',
      page: 15,
      url: null,
      timestamp: '2024-01-15T10:30:00Z',
      relevanceScore: 0.92,
      author: 'Dr. Jane Smith',
      type: 'document'
    },
    '2': {
      id: '2',
      title: 'Latest AI Research Breakthroughs',
      source: 'TechNews.com',
      content: 'Recent advances in transformer architecture have led to significant improvements in natural language processing capabilities. The new models demonstrate unprecedented performance in understanding context and generating human-like responses.',
      page: null,
      url: 'https://technews.com/ai-breakthroughs-2024',
      timestamp: '2024-01-20T14:22:00Z',
      relevanceScore: 0.88,
      author: 'Tech News Team',
      type: 'web'
    }
  };
  
  return citations[id] || null;
};

export default function CitationModal({ citationId, isOpen, onClose }: CitationModalProps) {
  const citation = getCitationData(citationId);

  const handleCopyToClipboard = async () => {
    if (!citation) return;
    
    const citationText = `"${citation.content}"\n\nSource: ${citation.title}${citation.author ? ` by ${citation.author}` : ''}${citation.page ? `, page ${citation.page}` : ''}${citation.url ? `\nURL: ${citation.url}` : ''}`;
    
    try {
      await navigator.clipboard.writeText(citationText);
      // You could add a toast notification here
    } catch (error) {
      console.error('Failed to copy to clipboard:', error);
    }
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'web':
        return <LinkIcon className="w-5 h-5" />;
      case 'document':
      default:
        return <DocumentTextIcon className="w-5 h-5" />;
    }
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'web':
        return 'text-green-600 dark:text-green-400 bg-green-100 dark:bg-green-900/30';
      case 'document':
      default:
        return 'text-blue-600 dark:text-blue-400 bg-blue-100 dark:bg-blue-900/30';
    }
  };

  if (!isOpen || !citation) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50"
        onClick={onClose}
      >
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.9, opacity: 0 }}
          className="bg-white dark:bg-gray-800 rounded-xl shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-hidden"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div className="p-6 border-b border-gray-200 dark:border-gray-700">
            <div className="flex items-start justify-between">
              <div className="flex items-start space-x-3">
                <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${getTypeColor(citation.type)}`}>
                  {getTypeIcon(citation.type)}
                </div>
                <div className="flex-1">
                  <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-1">
                    {citation.title}
                  </h2>
                  <div className="flex items-center space-x-4 text-sm text-gray-500 dark:text-gray-400">
                                         <span className="flex items-center space-x-1">
                       <span className={`px-2 py-1 rounded-full text-xs font-medium ${getTypeColor(citation.type)}`}>
                         {citation.type === 'web' ? 'Web' : 'Document'}
                       </span>
                     </span>
                    {citation.author && <span>by {citation.author}</span>}
                    {citation.page && <span>Page {citation.page}</span>}
                  </div>
                </div>
              </div>
              
              <div className="flex items-center space-x-2">
                <button
                  onClick={handleCopyToClipboard}
                  className="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
                  title="Copy citation"
                >
                  <ClipboardDocumentIcon className="w-5 h-5" />
                </button>
                <button
                  onClick={onClose}
                  className="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
                >
                  <XMarkIcon className="w-5 h-5" />
                </button>
              </div>
            </div>
          </div>

          {/* Content */}
          <div className="p-6 overflow-y-auto max-h-[60vh]">
            {/* Source Information */}
            <div className="mb-6">
              <h3 className="text-sm font-medium text-gray-900 dark:text-white mb-3">
                Source Information
              </h3>
              <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4 space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Source:</span>
                  <span className="text-sm font-medium text-gray-900 dark:text-white">
                    {citation.source}
                  </span>
                </div>
                {citation.url && (
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-600 dark:text-gray-400">URL:</span>
                    <a
                      href={citation.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-sm text-blue-600 dark:text-blue-400 hover:underline max-w-xs truncate"
                    >
                      {citation.url}
                    </a>
                  </div>
                )}
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Accessed:</span>
                  <span className="text-sm text-gray-900 dark:text-white">
                    {formatDate(citation.timestamp)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Relevance:</span>
                  <span className="text-sm text-gray-900 dark:text-white">
                    {(citation.relevanceScore * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            </div>

            {/* Citation Content */}
            <div>
              <h3 className="text-sm font-medium text-gray-900 dark:text-white mb-3">
                Cited Content
              </h3>
              <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                <blockquote className="text-gray-900 dark:text-white leading-relaxed">
                  "{citation.content}"
                </blockquote>
              </div>
            </div>
          </div>

          {/* Footer */}
          <div className="p-6 border-t border-gray-200 dark:border-gray-700 flex items-center justify-between">
            <div className="text-xs text-gray-500 dark:text-gray-400">
              Citation ID: {citation.id}
            </div>
            <div className="flex space-x-3">
              {citation.url && (
                <a
                  href={citation.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="px-4 py-2 text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 border border-blue-200 dark:border-blue-700 rounded-lg hover:bg-blue-50 dark:hover:bg-blue-900/20 transition-colors"
                >
                  Open Source
                </a>
              )}
              <button
                onClick={onClose}
                className="px-4 py-2 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
              >
                Close
              </button>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
} 