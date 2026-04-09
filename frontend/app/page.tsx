'use client'

import { useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { Loader2, Brain } from 'lucide-react'
import SignatureUploader from '@/components/SignatureUploader'
import ResultCard from '@/components/ResultCard'

interface VerificationResult {
  is_authentic: boolean
  similarity_score: number
  confidence: string
  difference_heatmap: string
  verdict: string
  inference_mode?: 'hybrid-neural' | 'classical-fallback' | string
  neural_score?: number | null
  classical_score?: number | null
}

type AppState = 'upload' | 'verifying' | 'result' | 'error'

/**
 * Main application page
 * Handles signature verification workflow
 */
export default function Home() {
  const [state, setState] = useState<AppState>('upload')
  const [genuineFile, setGenuineFile] = useState<File | null>(null)
  const [testFile, setTestFile] = useState<File | null>(null)
  const [genuinePreview, setGenuinePreview] = useState<string>('')
  const [testPreview, setTestPreview] = useState<string>('')
  const [result, setResult] = useState<VerificationResult | null>(null)
  const [error, setError] = useState<string>('')
  const [showWarmupBanner, setShowWarmupBanner] = useState(false)

  // Handle genuine signature selection
  const handleGenuineFileSelect = (file: File | null) => {
    if (file) {
      setGenuineFile(file)
      const reader = new FileReader()
      reader.onloadend = () => {
        setGenuinePreview(reader.result as string)
      }
      reader.readAsDataURL(file)
    } else {
      setGenuineFile(null)
      setGenuinePreview('')
    }
  }

  // Handle test signature selection
  const handleTestFileSelect = (file: File | null) => {
    if (file) {
      setTestFile(file)
      const reader = new FileReader()
      reader.onloadend = () => {
        setTestPreview(reader.result as string)
      }
      reader.readAsDataURL(file)
    } else {
      setTestFile(null)
      setTestPreview('')
    }
  }

  // Verify signatures
  const handleVerify = async () => {
    if (!genuineFile || !testFile) {
      setError('Please upload both signature images')
      return
    }

    setState('verifying')
    setError('')
    setShowWarmupBanner(false)

    const warmupTimer = setTimeout(() => {
      setShowWarmupBanner(true)
    }, 8000)

    try {
      const formData = new FormData()
      formData.append('genuine', genuineFile)
      formData.append('test', testFile)

      const response = await fetch('/api/verify', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || 'Verification failed')
      }

      const resultData = await response.json()
      setResult(resultData)
      setState('result')
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'An error occurred during verification'
      setError(errorMessage)
      setState('error')
    } finally {
      clearTimeout(warmupTimer)
      setShowWarmupBanner(false)
    }
  }

  // Reset to initial state
  const handleReset = () => {
    setGenuineFile(null)
    setTestFile(null)
    setGenuinePreview('')
    setTestPreview('')
    setResult(null)
    setError('')
    setState('upload')
  }

  const isVerifyDisabled = !genuineFile || !testFile || state !== 'upload'

  return (
    <main className="flex-1">
      {/* Header */}
      <header className="border-b border-dark-700 bg-gradient-to-r from-dark-800 via-dark-800 to-dark-900">
        <div className="container mx-auto px-4 py-12">
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-2 text-center"
          >
            <div className="flex items-center justify-center gap-3 mb-4">
              <motion.div
                animate={{
                  rotate: [0, 360],
                }}
                transition={{
                  duration: 4,
                  repeat: Infinity,
                  ease: 'linear',
                }}
              >
                <Brain className="w-8 h-8 text-primary-500" />
              </motion.div>
              <h1 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-primary-400 via-primary-500 to-blue-400 bg-clip-text text-transparent">
                Signature Authentication System
              </h1>
            </div>
            <p className="text-gray-400 text-lg max-w-2xl mx-auto">
              Hybrid Siamese CNN verification with classical fallback scoring
            </p>
          </motion.div>
        </div>
      </header>

      {/* Main Content */}
      <div className="container mx-auto px-4 py-12 flex-1">
        <AnimatePresence mode="wait">
          {/* Upload State */}
          {state === 'upload' && (
            <motion.div
              key="upload"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
              className="max-w-4xl mx-auto"
            >
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
                {/* Genuine Signature Upload */}
                <SignatureUploader
                  label="Upload Genuine Signature (Reference)"
                  onFileSelect={handleGenuineFileSelect}
                  selectedFile={genuineFile ?? undefined}
                  previewUrl={genuinePreview}
                />

                {/* Test Signature Upload */}
                <SignatureUploader
                  label="Upload Test Signature (To Verify)"
                  onFileSelect={handleTestFileSelect}
                  selectedFile={testFile ?? undefined}
                  previewUrl={testPreview}
                />
              </div>

              {/* Error Message */}
              <AnimatePresence>
                {error && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    className="mb-6 p-4 bg-red-900/20 border border-red-700/50 rounded-lg text-red-300 text-sm"
                  >
                    {error}
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Verify Button */}
              <motion.div
                className="flex justify-center"
                whileHover={!isVerifyDisabled ? { scale: 1.02 } : {}}
                whileTap={!isVerifyDisabled ? { scale: 0.98 } : {}}
              >
                <button
                  onClick={handleVerify}
                  disabled={isVerifyDisabled}
                  className="btn-primary text-lg px-12 py-3 flex items-center gap-2"
                >
                  <Brain className="w-5 h-5" />
                  Verify Signature
                </button>
              </motion.div>

              {/* Info Section - Key Stats */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="mt-12 grid grid-cols-1 md:grid-cols-3 gap-4"
              >
                <motion.div
                  whileHover={{ translateY: -4 }}
                  className="card text-center bg-gradient-to-br from-purple-900/20 to-purple-800/10 border-purple-700/30 hover:border-purple-600/50 transition-colors"
                >
                  <div className="text-4xl md:text-5xl font-bold mb-3 text-transparent bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text">
                    HY
                  </div>
                  <div className="text-sm text-gray-300 font-medium">
                    Hybrid Neural Inference
                  </div>
                </motion.div>

                <motion.div
                  whileHover={{ translateY: -4 }}
                  className="card text-center bg-gradient-to-br from-emerald-900/20 to-emerald-800/10 border-emerald-700/30 hover:border-emerald-600/50 transition-colors"
                >
                  <div className="text-4xl md:text-5xl font-bold mb-3 text-transparent bg-gradient-to-r from-emerald-400 to-cyan-400 bg-clip-text">
                    DL
                  </div>
                  <div className="text-sm text-gray-300 font-medium">
                    Siamese + Classical Blend
                  </div>
                </motion.div>

                <motion.div
                  whileHover={{ translateY: -4 }}
                  className="card text-center bg-gradient-to-br from-blue-900/20 to-blue-800/10 border-blue-700/30 hover:border-blue-600/50 transition-colors"
                >
                  <div className="text-4xl md:text-5xl font-bold mb-3 text-transparent bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text">
                    2x
                  </div>
                  <div className="text-sm text-gray-300 font-medium">
                    Mode-Aware Thresholding
                  </div>
                </motion.div>
              </motion.div>
            </motion.div>
          )}

          {/* Verifying State */}
          {state === 'verifying' && (
            <motion.div
              key="verifying"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="max-w-2xl mx-auto flex flex-col items-center justify-center gap-6 py-20"
            >
              <motion.div
                animate={{
                  rotate: 360,
                  scale: [1, 1.1, 1],
                }}
                transition={{
                  rotate: {
                    duration: 2,
                    repeat: Infinity,
                    ease: 'linear',
                  },
                  scale: {
                    duration: 2,
                    repeat: Infinity,
                    ease: 'easeInOut',
                  },
                }}
              >
                <Loader2 className="w-16 h-16 text-primary-500" />
              </motion.div>

              <div className="text-center space-y-2">
                <h2 className="text-2xl font-bold">Analyzing Signatures</h2>
                <p className="text-gray-400 text-lg">
                  Running hybrid neural scoring with robustness checks...
                </p>
              </div>

              <AnimatePresence>
                {showWarmupBanner && (
                  <motion.div
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -8 }}
                    className="w-full rounded-lg border border-amber-500/40 bg-amber-500/10 px-4 py-3 text-sm text-amber-200"
                  >
                    Model is warming up, this may take up to 20 seconds on first request.
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Progress Dots */}
              <div className="flex gap-2">
                {[0, 1, 2].map((i) => (
                  <motion.div
                    key={i}
                    initial={{ scale: 0.5, opacity: 0.3 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{
                      duration: 0.6,
                      repeat: Infinity,
                      delay: i * 0.2,
                    }}
                    className="w-3 h-3 rounded-full bg-primary-400"
                  />
                ))}
              </div>
            </motion.div>
          )}

          {/* Result State */}
          {state === 'result' && result && (
            <motion.div
              key="result"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="max-w-4xl mx-auto"
            >
              <ResultCard
                result={result}
                genuineImageUrl={genuinePreview}
                testImageUrl={testPreview}
                onReset={handleReset}
              />
            </motion.div>
          )}

          {/* Error State */}
          {state === 'error' && (
            <motion.div
              key="error"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className="max-w-2xl mx-auto"
            >
              <div className="card border-red-700/50 bg-red-900/10">
                <div className="space-y-4">
                  <h2 className="text-2xl font-bold text-red-400">Verification Failed</h2>
                  <p className="text-gray-300">{error}</p>
                  <button
                    onClick={handleReset}
                    className="btn-primary"
                  >
                    Try Again
                  </button>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Footer */}
      <footer className="border-t border-dark-700 bg-dark-800/50 py-8 mt-12">
        <div className="container mx-auto px-4 text-center">
          <p className="text-gray-500 text-sm">
            Signature Authentication System © 2024 | AI-Powered Forgery Detection
          </p>
        </div>
      </footer>
    </main>
  )
}
