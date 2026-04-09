'use client'

import { motion } from 'framer-motion'
import { CheckCircle, XCircle, AlertCircle, Eye, EyeOff } from 'lucide-react'
import { useState } from 'react'
import SimilarityMeter from './SimilarityMeter'

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

interface ResultCardProps {
  result: VerificationResult
  genuineImageUrl: string
  testImageUrl: string
  onReset: () => void
}

/**
 * Result card displaying verification results with side-by-side comparison
 * and optional difference heatmap overlay
 */
export default function ResultCard({
  result,
  genuineImageUrl,
  testImageUrl,
  onReset,
}: ResultCardProps) {
  const [showHeatmap, setShowHeatmap] = useState(false)
  const percentage = Math.round(result.similarity_score * 100)
  const mode = result.inference_mode ?? 'classical-fallback'

  const thresholds = mode === 'hybrid-neural'
    ? { genuine: 90, suspicious: 80 }
    : { genuine: 93, suspicious: 82 }

  const getVerdictIcon = (verdict: string) => {
    switch (verdict.toLowerCase()) {
      case 'genuine':
        return <CheckCircle className="w-12 h-12 text-green-400" />
      case 'forged':
        return <XCircle className="w-12 h-12 text-red-400" />
      case 'suspicious':
        return <AlertCircle className="w-12 h-12 text-yellow-400" />
      default:
        return null
    }
  }

  const getVerdictBadgeClass = (verdict: string) => {
    switch (verdict.toLowerCase()) {
      case 'genuine':
        return 'badge-success'
      case 'forged':
        return 'badge-danger'
      case 'suspicious':
        return 'badge-warning'
      default:
        return ''
    }
  }

  const getScoreBarColorClass = (score: number) => {
    if (score >= 0.9) return 'bg-green-500'
    if (score >= 0.8) return 'bg-amber-500'
    return 'bg-red-500'
  }

  const getScoreTextColorClass = (score: number) => {
    if (score >= 0.9) return 'text-green-400'
    if (score >= 0.8) return 'text-amber-400'
    return 'text-red-400'
  }

  const renderScoreBar = (label: string, score: number) => {
    const width = Math.max(0, Math.min(100, score * 100))

    return (
      <div className="space-y-2" key={label}>
        <div className="flex items-center justify-between text-sm">
          <span className="text-gray-300">{label}</span>
          <span className={`font-semibold ${getScoreTextColorClass(score)}`}>
            {Math.round(width)}%
          </span>
        </div>
        <div className="h-2.5 w-full overflow-hidden rounded-full bg-dark-700">
          <div
            className={`h-full ${getScoreBarColorClass(score)}`}
            style={{ width: `${width}%` }}
          />
        </div>
      </div>
    )
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 20 }}
      transition={{ duration: 0.5 }}
      className="w-full space-y-8"
    >
      {/* Verdict Section */}
      <motion.div
        initial={{ scale: 0.8, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ delay: 0.2, type: 'spring', stiffness: 100 }}
        className="card text-center space-y-4"
      >
        <motion.div
          animate={{
            scale: [1, 1.05, 1],
          }}
          transition={{
            duration: 2,
            repeat: Infinity,
            repeatType: 'loop',
          }}
        >
          {getVerdictIcon(result.verdict)}
        </motion.div>

        <div className="space-y-2">
          <h2 className="text-3xl font-bold">
            {result.verdict === 'Genuine' && '✓ GENUINE'}
            {result.verdict === 'Forged' && '✗ FORGED'}
            {result.verdict === 'Suspicious' && '⚠ SUSPICIOUS'}
          </h2>
          <div className={`badge ${getVerdictBadgeClass(result.verdict)} justify-center w-fit mx-auto`}>
            {result.confidence} Confidence
          </div>

          {(typeof result.neural_score === 'number' || typeof result.classical_score === 'number') && (
            <div className="mx-auto mt-4 w-full max-w-xl space-y-3 rounded-lg border border-dark-700 bg-dark-800/40 p-4 text-left">
              <div className="text-xs font-semibold uppercase tracking-wide text-gray-400">
                Confidence Indicators
              </div>
              {typeof result.neural_score === 'number' && renderScoreBar('Neural Score', result.neural_score)}
              {typeof result.classical_score === 'number' && renderScoreBar('Classical Score', result.classical_score)}
            </div>
          )}
        </div>
      </motion.div>

      {/* Similarity Meter */}
      <SimilarityMeter
        score={result.similarity_score * 100}
        inferenceMode={mode}
        isVisible={true}
      />

      {/* Image Comparison */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="card"
      >
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <span className="text-primary-400">→</span>
          Signature Comparison
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Genuine Signature */}
          <div className="space-y-2">
            <div className="text-sm text-gray-400 font-semibold">GENUINE (Reference)</div>
            <div className="relative bg-dark-700 rounded-lg overflow-hidden border border-dark-600 aspect-[3/4]">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={genuineImageUrl}
                alt="Genuine signature"
                className="w-full h-full object-cover"
              />
            </div>
          </div>

          {/* Test Signature */}
          <div className="space-y-2">
            <div className="text-sm text-gray-400 font-semibold">TEST (Verification)</div>
            <div className="relative bg-dark-700 rounded-lg overflow-hidden border border-dark-600 aspect-[3/4]">
              {showHeatmap && result.difference_heatmap ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={`data:image/png;base64,${result.difference_heatmap}`}
                  alt="Difference heatmap"
                  className="w-full h-full object-cover"
                />
              ) : (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={testImageUrl}
                  alt="Test signature"
                  className="w-full h-full object-cover"
                />
              )}
            </div>
          </div>
        </div>

        {/* Heatmap Toggle */}
        {result.difference_heatmap && (
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => setShowHeatmap(!showHeatmap)}
            className="btn-secondary mt-4 w-full flex items-center justify-center gap-2"
          >
            {showHeatmap ? (
              <>
                <EyeOff className="w-4 h-4" />
                Hide Difference Map
              </>
            ) : (
              <>
                <Eye className="w-4 h-4" />
                Show Difference Map
              </>
            )}
          </motion.button>
        )}
      </motion.div>

      {/* Details Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6 }}
        className="card space-y-4"
      >
        <h3 className="text-lg font-semibold">Verification Details</h3>

        <div className="inline-flex items-center rounded-full border border-primary-500/40 bg-primary-500/10 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-primary-300">
          Inference Mode: {mode === 'hybrid-neural' ? 'Hybrid Neural' : 'Classical Fallback'}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-2">
            <div className="text-sm text-gray-400">Similarity Score</div>
            <div className="text-2xl font-bold text-primary-400">{percentage}%</div>
          </div>

          <div className="space-y-2">
            <div className="text-sm text-gray-400">Confidence Level</div>
            <div className="text-2xl font-bold">
              {result.confidence === 'High' && (
                <span className="text-green-400">HIGH</span>
              )}
              {result.confidence === 'Medium' && (
                <span className="text-yellow-400">MEDIUM</span>
              )}
              {result.confidence === 'Low' && (
                <span className="text-red-400">LOW</span>
              )}
            </div>
          </div>

          <div className="space-y-2">
            <div className="text-sm text-gray-400">Authentication Status</div>
            <div className="text-2xl font-bold">
              {result.is_authentic ? (
                <span className="text-green-400">AUTHENTIC</span>
              ) : (
                <span className="text-red-400">NOT AUTHENTIC</span>
              )}
            </div>
          </div>

          <div className="space-y-2">
            <div className="text-sm text-gray-400">Verdict</div>
            <div className="text-2xl font-bold">{result.verdict.toUpperCase()}</div>
          </div>

          {typeof result.neural_score === 'number' && (
            <div className="space-y-2">
              <div className="text-sm text-gray-400">Neural Score</div>
              <div className="text-2xl font-bold text-violet-300">
                {Math.round(result.neural_score * 100)}%
              </div>
            </div>
          )}

          {typeof result.classical_score === 'number' && (
            <div className="space-y-2">
              <div className="text-sm text-gray-400">Classical Score</div>
              <div className="text-2xl font-bold text-cyan-300">
                {Math.round(result.classical_score * 100)}%
              </div>
            </div>
          )}
        </div>

        {/* Score Range Info */}
        <div className="border-t border-dark-700 pt-4 mt-4">
          <div className="text-xs text-gray-400 space-y-2">
            <p>
              <span className="text-green-400">≥ {thresholds.genuine}%</span>
              {' '}→ GENUINE (High Confidence)
            </p>
            <p>
              <span className="text-yellow-400">{thresholds.suspicious}-{thresholds.genuine - 1}%</span>
              {' '}→ SUSPICIOUS (Medium Confidence)
            </p>
            <p>
              <span className="text-red-400">&lt; {thresholds.suspicious}%</span>
              {' '}→ FORGED (High Confidence)
            </p>
          </div>
        </div>
      </motion.div>

      {/* Action Buttons */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.8 }}
        className="flex gap-4 justify-center"
      >
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={onReset}
          className="btn-primary"
        >
          Try Another Verification
        </motion.button>
      </motion.div>
    </motion.div>
  )
}
