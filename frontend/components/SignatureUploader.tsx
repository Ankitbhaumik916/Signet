'use client'

import { useCallback, useMemo } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, X } from 'lucide-react'
import { motion } from 'framer-motion'

interface SignatureUploaderProps {
  label: string
  onFileSelect: (file: File | null) => void
  selectedFile?: File | null
  previewUrl?: string
  isProcessing?: boolean
}

/**
 * Drag & drop file uploader component for signature images
 * Supports click-to-upload and drag & drop
 */
export default function SignatureUploader({
  label,
  onFileSelect,
  selectedFile,
  previewUrl,
  isProcessing = false,
}: SignatureUploaderProps) {
  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (acceptedFiles.length > 0) {
        const file = acceptedFiles[0]
        // Validate file type
        if (file.type.startsWith('image/')) {
          onFileSelect(file)
        } else {
          alert('Please upload a valid image file')
        }
      }
    },
    [onFileSelect]
  )

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.gif', '.bmp', '.webp'],
    },
    disabled: isProcessing || selectedFile !== undefined,
    maxFiles: 1,
  })

  const handleRemove = (e: React.MouseEvent) => {
    e.stopPropagation()
    onFileSelect(null)
  }

  // Get file size in MB
  const fileSize = useMemo(() => {
    if (!selectedFile) return ''
    return (selectedFile.size / 1024 / 1024).toFixed(2)
  }, [selectedFile])

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-3"
    >
      <label className="block text-sm font-semibold text-gray-200">
        {label}
      </label>

      {selectedFile && previewUrl ? (
        // Preview mode
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="card space-y-4"
          role="region"
        >
          {/* Preview Image */}
          <div className="relative bg-dark-700 rounded-lg overflow-hidden border border-primary-600/30 aspect-[3/4] flex items-center justify-center">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={previewUrl}
              alt="Signature preview"
              className="w-full h-full object-cover"
            />
          </div>

          {/* File Info */}
          <div className="space-y-2 text-sm text-gray-300">
            <p>
              <span className="text-gray-400">Name:</span> {selectedFile.name}
            </p>
            <p>
              <span className="text-gray-400">Size:</span> {fileSize} MB
            </p>
            <p>
              <span className="text-gray-400">Type:</span> {selectedFile.type}
            </p>
          </div>

          {/* Remove Button */}
          {!isProcessing && (
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={handleRemove}
              className="btn-secondary w-full flex items-center justify-center gap-2"
            >
              <X className="w-4 h-4" />
              Choose Different Image
            </motion.button>
          )}
        </motion.div>
      ) : (
        // Upload zone
        <div
          {...getRootProps()}
          className={`
            relative p-8 rounded-lg border-2 border-dashed transition-all duration-200 cursor-pointer
            flex flex-col items-center justify-center gap-4 min-h-[200px]
            ${
              isDragActive
                ? 'border-primary-500 bg-primary-900/20'
                : 'border-dark-600 bg-dark-800 hover:bg-dark-700 hover:border-dark-500'
            }
            ${isProcessing ? 'opacity-50 cursor-not-allowed' : ''}
          `}
        >
          <input {...getInputProps()} />

          <motion.div
            animate={isDragActive ? { scale: 1.2 } : { scale: 1 }}
            className="p-3 bg-primary-900/30 rounded-lg"
          >
            <Upload className="w-6 h-6 text-primary-400" />
          </motion.div>

          <div className="text-center">
            <p className="font-semibold text-gray-100">
              {isDragActive ? 'Drop your signature here' : 'Drag & drop your signature here'}
            </p>
            <p className="text-sm text-gray-400 mt-1">
              or click to select an image
            </p>
          </div>

          <div className="text-xs text-gray-500 text-center">
            Supported formats: JPG, PNG, GIF, BMP, WebP
          </div>
        </div>
      )}
    </motion.div>
  )
}
