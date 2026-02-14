'use client';

import React, { useState, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  DocumentArrowUpIcon, 
  XMarkIcon,
  CloudArrowUpIcon,
  CheckCircleIcon,
  ExclamationCircleIcon
} from '@heroicons/react/24/outline';
import { uploadDocument } from '@/lib/api';

interface DocumentUploadProps {
  isOpen: boolean;
  onClose: () => void;
  onUploadSuccess?: (uploadedFiles: { filename: string; document_id: string }[]) => void;
}

interface UploadFile {
  id: string;
  file: File;
  progress: number;
  status: 'pending' | 'uploading' | 'success' | 'error';
  error?: string;
  uploadResult?: {
    document_id: string;
    filename: string;
  };
}

export default function DocumentUpload({ isOpen, onClose, onUploadSuccess }: DocumentUploadProps) {
  const [files, setFiles] = useState<UploadFile[]>([]);
  const [isDragOver, setIsDragOver] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const allowedTypes = [
    'application/pdf',
    'application/msword',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'text/plain',
    'text/markdown',
    'image/jpeg',
    'image/png',
    'image/gif'
  ];

  const maxFileSize = 50 * 1024 * 1024; // 50MB

  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);

    const droppedFiles = Array.from(e.dataTransfer.files);
    handleFiles(droppedFiles);
  }, []);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const selectedFiles = Array.from(e.target.files);
      handleFiles(selectedFiles);
    }
  }, []);

  const handleFiles = useCallback((newFiles: File[]) => {
    const validFiles = newFiles.filter(file => {
      if (!allowedTypes.includes(file.type)) {
        alert(`File type ${file.type} is not supported`);
        return false;
      }
      if (file.size > maxFileSize) {
        alert(`File ${file.name} is too large. Maximum size is 50MB`);
        return false;
      }
      return true;
    });

    const uploadFiles: UploadFile[] = validFiles.map(file => ({
      id: Date.now().toString() + Math.random().toString(36).substring(2),
      file,
      progress: 0,
      status: 'pending'
    }));

    setFiles(prev => [...prev, ...uploadFiles]);
  }, []);

  const uploadFile = async (uploadFile: UploadFile) => {
    try {
      setFiles(prev => prev.map(f => 
        f.id === uploadFile.id 
          ? { ...f, status: 'uploading', progress: 0 }
          : f
      ));

      // Simulate upload progress
      const progressInterval = setInterval(() => {
        setFiles(prev => prev.map(f => {
          if (f.id === uploadFile.id && f.progress < 90) {
            return { ...f, progress: f.progress + 10 };
          }
          return f;
        }));
      }, 200);

      const result = await uploadDocument(uploadFile.file);

      clearInterval(progressInterval);

      if (result.success) {
        setFiles(prev => prev.map(f => 
          f.id === uploadFile.id 
            ? { 
                ...f, 
                status: 'success', 
                progress: 100,
                uploadResult: {
                  document_id: result.document_id || '',
                  filename: result.filename || uploadFile.file.name
                }
              }
            : f
        ));
      } else {
        throw new Error(result.message || 'Upload failed');
      }
    } catch (error) {
      setFiles(prev => prev.map(f => 
        f.id === uploadFile.id 
          ? { 
              ...f, 
              status: 'error', 
              progress: 0,
              error: error instanceof Error ? error.message : 'Upload failed'
            }
          : f
      ));
    }
  };

  const uploadAllFiles = async () => {
    setIsUploading(true);
    const pendingFiles = files.filter(f => f.status === 'pending');
    
    try {
      await Promise.all(pendingFiles.map(uploadFile));
      
      // Wait a bit for state updates to complete
      setTimeout(() => {
        setFiles(currentFiles => {
          const successfulFiles = currentFiles.filter(f => f.status === 'success' && f.uploadResult);
          const hasErrors = currentFiles.some(f => f.status === 'error');
          const hasPending = currentFiles.some(f => f.status === 'pending');
          
          // If all uploads completed and we have successful uploads
          if (successfulFiles.length > 0 && !hasErrors && !hasPending) {
            // Notify parent about successful uploads
            if (onUploadSuccess) {
              const uploadResults = successfulFiles.map(f => f.uploadResult!);
              onUploadSuccess(uploadResults);
            }
            
            // Auto-close modal after a brief delay
            setTimeout(() => {
              setFiles([]);
              onClose();
            }, 1000);
          }
          
          return currentFiles;
        });
      }, 500);
    } finally {
      setIsUploading(false);
    }
  };

  const removeFile = (fileId: string) => {
    setFiles(prev => prev.filter(f => f.id !== fileId));
  };

  const clearAllFiles = () => {
    setFiles([]);
  };

  const handleClose = () => {
    if (!isUploading) {
      setFiles([]);
      onClose();
    }
  };

  const formatFileSize = (bytes: number) => {
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    if (bytes === 0) return '0 Bytes';
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
  };

  const getStatusIcon = (status: UploadFile['status']) => {
    switch (status) {
      case 'success':
        return <CheckCircleIcon className="w-5 h-5 text-green-500" />;
      case 'error':
        return <ExclamationCircleIcon className="w-5 h-5 text-red-500" />;
      case 'uploading':
        return (
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
            className="w-5 h-5 border-2 border-blue-500 border-t-transparent rounded-full"
          />
        );
      default:
        return <DocumentArrowUpIcon className="w-5 h-5 text-gray-400" />;
    }
  };

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50"
        onClick={handleClose}
      >
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.9, opacity: 0 }}
          className="bg-white dark:bg-gray-800 rounded-xl shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-hidden"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div className="p-6 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-blue-100 dark:bg-blue-900/30 rounded-lg flex items-center justify-center">
                <CloudArrowUpIcon className="w-6 h-6 text-blue-600 dark:text-blue-400" />
              </div>
              <div>
                <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                  Upload Documents
                </h2>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  Add documents to your knowledge base
                </p>
              </div>
            </div>
            <button
              onClick={handleClose}
              disabled={isUploading}
              className="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 disabled:opacity-50"
            >
              <XMarkIcon className="w-6 h-6" />
            </button>
          </div>

          {/* Content */}
          <div className="p-6 overflow-y-auto max-h-[60vh]">
            {/* Drop Zone */}
            <div
              onDragEnter={handleDragEnter}
              onDragLeave={handleDragLeave}
              onDragOver={handleDragOver}
              onDrop={handleDrop}
              className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
                isDragOver
                  ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                  : 'border-gray-300 dark:border-gray-600'
              }`}
            >
              <CloudArrowUpIcon className="w-12 h-12 text-gray-400 mx-auto mb-4" />
              <div className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                Drag and drop files here
              </div>
              <div className="text-sm text-gray-500 dark:text-gray-400 mb-4">
                or click to browse files
              </div>
              <button
                onClick={() => fileInputRef.current?.click()}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                Choose Files
              </button>
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept={allowedTypes.join(',')}
                onChange={handleFileInput}
                className="hidden"
              />
            </div>

            {/* Supported formats */}
            <div className="mt-4 text-xs text-gray-500 dark:text-gray-400">
              Supported formats: PDF, DOC, DOCX, TXT, MD, JPG, PNG, GIF (max 50MB each)
            </div>

            {/* File List */}
            {files.length > 0 && (
              <div className="mt-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-medium text-gray-900 dark:text-white">
                    Files ({files.length})
                  </h3>
                  <button
                    onClick={clearAllFiles}
                    disabled={isUploading}
                    className="text-sm text-red-600 hover:text-red-700 disabled:opacity-50"
                  >
                    Clear All
                  </button>
                </div>

                <div className="space-y-3">
                  {files.map((file) => (
                    <div
                      key={file.id}
                      className="flex items-center space-x-3 p-3 bg-gray-50 dark:bg-gray-700 rounded-lg"
                    >
                      {getStatusIcon(file.status)}
                      
                      <div className="flex-1 min-w-0">
                        <div className="text-sm font-medium text-gray-900 dark:text-white truncate">
                          {file.file.name}
                        </div>
                        <div className="text-xs text-gray-500 dark:text-gray-400">
                          {formatFileSize(file.file.size)}
                        </div>
                        
                        {file.status === 'uploading' && (
                          <div className="mt-2">
                            <div className="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-2">
                              <div
                                className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                                style={{ width: `${file.progress}%` }}
                              />
                            </div>
                          </div>
                        )}
                        
                        {file.status === 'error' && file.error && (
                          <div className="text-xs text-red-600 dark:text-red-400 mt-1">
                            {file.error}
                          </div>
                        )}
                      </div>

                      {file.status !== 'uploading' && (
                        <button
                          onClick={() => removeFile(file.id)}
                          className="p-1 text-gray-400 hover:text-red-600"
                        >
                          <XMarkIcon className="w-4 h-4" />
                        </button>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Footer */}
          {files.length > 0 && (
            <div className="p-6 border-t border-gray-200 dark:border-gray-700 flex items-center justify-between">
              <div className="text-sm text-gray-500 dark:text-gray-400">
                {files.filter(f => f.status === 'success').length} of {files.length} uploaded
              </div>
              <div className="flex space-x-3">
                <button
                  onClick={handleClose}
                  disabled={isUploading}
                  className="px-4 py-2 text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white disabled:opacity-50"
                >
                  Cancel
                </button>
                <button
                  onClick={uploadAllFiles}
                  disabled={isUploading || files.filter(f => f.status === 'pending').length === 0}
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isUploading ? 'Uploading...' : 'Upload All'}
                </button>
              </div>
            </div>
          )}
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
} 