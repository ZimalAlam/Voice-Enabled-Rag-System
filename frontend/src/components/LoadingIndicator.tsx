'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { CpuChipIcon } from '@heroicons/react/24/outline';

export default function LoadingIndicator() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="flex justify-start"
    >
      <div className="flex max-w-[80%] items-start space-x-3">
        {/* Avatar */}
        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-r from-purple-500 to-blue-600 flex items-center justify-center">
          <CpuChipIcon className="w-5 h-5 text-white" />
        </div>

        {/* Loading Content */}
        <div className="flex flex-col space-y-2">
          {/* Loading Bubble */}
          <div className="px-4 py-3 bg-white dark:bg-gray-700 text-gray-900 dark:text-white border border-gray-200 dark:border-gray-600 rounded-2xl">
            <div className="flex items-center space-x-2">
              <span className="text-gray-600 dark:text-gray-400">Thinking</span>
              <div className="flex space-x-1">
                {[0, 1, 2].map((i) => (
                  <motion.div
                    key={i}
                    className="w-2 h-2 bg-blue-500 rounded-full"
                    animate={{
                      scale: [1, 1.2, 1],
                      opacity: [0.7, 1, 0.7],
                    }}
                    transition={{
                      duration: 1.5,
                      repeat: Infinity,
                      delay: i * 0.2,
                      ease: "easeInOut",
                    }}
                  />
                ))}
              </div>
            </div>
          </div>

          {/* Processing Status */}
          <div className="text-xs text-gray-500 dark:text-gray-400">
            <motion.div
              animate={{
                opacity: [0.5, 1, 0.5],
              }}
              transition={{
                duration: 2,
                repeat: Infinity,
                ease: "easeInOut",
              }}
            >
              🔍 Searching documents and web sources...
            </motion.div>
          </div>
        </div>
      </div>
    </motion.div>
  );
} 