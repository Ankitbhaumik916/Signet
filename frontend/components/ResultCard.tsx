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
        return <CheckCircle className="w-12 h-12" style={{ color: 'var(--success)' }} />
      case 'forged':
        return <XCircle className="w-12 h-12" style={{ color: 'var(--danger)' }} />
      case 'suspicious':
        return <AlertCircle className="w-12 h-12" style={{ color: 'var(--warning)' }} />
      default:
        return null
    }
  }

  const getVerdictBadgeClass = (verdict: string) => {
    switch (verdict.toLowerCase()) {
      case 'genuine':
        return { bg: 'rgba(0, 229, 160, 0.1)', border: 'rgba(0, 229, 160, 0.3)', color: 'var(--success)' }
      case 'forged':
        return { bg: 'rgba(255, 74, 106, 0.1)', border: 'rgba(255, 74, 106, 0.3)', color: 'var(--danger)' }
      case 'suspicious':
        return { bg: 'rgba(255, 184, 0, 0.1)', border: 'rgba(255, 184, 0, 0.3)', color: 'var(--warning)' }
      default:
        return { bg: 'transparent', border: 'var(--border)', color: 'var(--text)' }
    }
  }

  const getScoreColor = (score: number) => {
    if (score >= 0.9) return 'var(--success)'
    if (score >= 0.8) return 'var(--warning)'
    return 'var(--danger)'
  }

  const renderScoreBar = (label: string, score: number) => {
    const width = Math.max(0, Math.min(100, score * 100))
    const color = getScoreColor(score)

    return (
      <div className="space-y-2" key={label}>
        <div className="flex items-center justify-between text-sm">
          <span style={{ color: 'var(--text)' }}>{label}</span>
          <span className="font-semibold" style={{ color }}>
            {Math.round(width)}%
          </span>
        </div>
        <div className="h-2.5 w-full overflow-hidden rounded-full" style={{ backgroundColor: 'var(--card)' }}>
          <div
            className="h-full"
            style={{ width: `${width}%`, backgroundColor: color }}
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
        className="p-6 rounded-xl border text-center space-y-4"
        style={{ backgroundColor: 'var(--card)', borderColor: 'var(--border)' }}
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
          <h2 className="text-3xl font-bold" style={{ color: 'var(--text)' }}>
            {result.verdict === 'Genuine' && '✓ GENUINE'}
            {result.verdict === 'Forged' && '✗ FORGED'}
            {result.verdict === 'Suspicious' && '⚠ SUSPICIOUS'}
          </h2>
          <div 
            className="inline-flex items-center px-3 py-1 rounded-full text-sm font-semibold border mx-auto"
            style={{ ...getVerdictBadgeClass(result.verdict), border: '1px solid ' + getVerdictBadgeClass(result.verdict).border }}
          >
            {result.confidence} Confidence
          </div>

          {(typeof result.neural_score === 'number' || typeof result.classical_score === 'number') && (
            <div className="mx-auto mt-4 w-full max-w-xl space-y-3 rounded-lg border p-4 text-left" style={{ backgroundColor: 'var(--card2)', borderColor: 'var(--border)' }}>
              <div className="text-xs font-semibold uppercase tracking-wide" style={{ color: 'var(--muted)' }}>
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
        className="p-6 rounded-xl border"
        style={{ backgroundColor: 'var(--card)', borderColor: 'var(--border)' }}
      >
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2" style={{ color: 'var(--text)' }}>
          <span style={{ color: 'var(--accent)' }}>→</span>
          Signature Comparison
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Genuine Signature */}
          <div className="space-y-2">
            <div className="text-sm font-semibold" style={{ color: 'var(--muted)' }}>GENUINE (Reference)</div>
            <div className="relative rounded-lg overflow-hidden border aspect-[3/4]" style={{ backgroundColor: 'var(--surface)', borderColor: 'var(--border)' }}>
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
            <div className="text-sm font-semibold" style={{ color: 'var(--muted)' }}>TEST (Verification)</div>
            <div className="relative rounded-lg overflow-hidden border aspect-[3/4]" style={{ backgroundColor: 'var(--surface)', borderColor: 'var(--border)' }}>
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
            className="mt-4 w-full flex items-center justify-center gap-2 px-6 py-3 rounded-lg font-semibold transition-all duration-200 border"
            style={{ backgroundColor: 'var(--surface)', borderColor: 'var(--border2)', color: 'var(--text)' }}
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
        className="p-6 rounded-xl border space-y-4"
        style={{ backgroundColor: 'var(--card)', borderColor: 'var(--border)' }}
      >
        <h3 className="text-lg font-semibold" style={{ color: 'var(--text)' }}>Verification Details</h3>

        <div 
          className="inline-flex items-center rounded-full border px-3 py-1 text-xs font-semibold uppercase tracking-wide"
          style={{ backgroundColor: 'var(--accent-dim)', borderColor: 'var(--border2)', color: 'var(--accent)' }}
        >
          Inference Mode: {mode === 'hybrid-neural' ? 'Hybrid Neural' : 'Classical Fallback'}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-2">
            <div className="text-sm" style={{ color: 'var(--muted)' }}>Similarity Score</div>
            <div className="text-2xl font-bold" style={{ color: 'var(--accent)' }}>{percentage}%</div>
          </div>

          <div className="space-y-2">
            <div className="text-sm" style={{ color: 'var(--muted)' }}>Confidence Level</div>
            <div className="text-2xl font-bold">
              {result.confidence === 'High' && (
                <span style={{ color: 'var(--success)' }}>HIGH</span>
              )}
              {result.confidence === 'Medium' && (
                <span style={{ color: 'var(--warning)' }}>MEDIUM</span>
              )}
              {result.confidence === 'Low' && (
                <span style={{ color: 'var(--danger)' }}>LOW</span>
              )}
            </div>
          </div>

          <div className="space-y-2">
            <div className="text-sm" style={{ color: 'var(--muted)' }}>Authentication Status</div>
            <div className="text-2xl font-bold">
              {result.is_authentic ? (
                <span style={{ color: 'var(--success)' }}>AUTHENTIC</span>
              ) : (
                <span style={{ color: 'var(--danger)' }}>NOT AUTHENTIC</span>
              )}
            </div>
          </div>

          <div className="space-y-2">
            <div className="text-sm" style={{ color: 'var(--muted)' }}>Verdict</div>
            <div className="text-2xl font-bold" style={{ color: 'var(--text)' }}>{result.verdict.toUpperCase()}</div>
          </div>

          {typeof result.neural_score === 'number' && (
            <div className="space-y-2">
              <div className="text-sm" style={{ color: 'var(--muted)' }}>Neural Score</div>
              <div className="text-2xl font-bold" style={{ color: '#a78bfa' }}>
                {Math.round(result.neural_score * 100)}%
              </div>
            </div>
          )}

          {typeof result.classical_score === 'number' && (
            <div className="space-y-2">
              <div className="text-sm" style={{ color: 'var(--muted)' }}>Classical Score</div>
              <div className="text-2xl font-bold" style={{ color: '#06b6d4' }}>
                {Math.round(result.classical_score * 100)}%
              </div>
            </div>
          )}
        </div>

        {/* Score Range Info */}
        <div className="border-t pt-4 mt-4" style={{ borderColor: 'var(--border)' }}>
          <div className="text-xs space-y-2" style={{ color: 'var(--muted)' }}>
            <p>
              <span style={{ color: 'var(--success)' }}>≥ {thresholds.genuine}%</span>
              {' '}→ GENUINE (High Confidence)
            </p>
            <p>
              <span style={{ color: 'var(--warning)' }}>{thresholds.suspicious}-{thresholds.genuine - 1}%</span>
              {' '}→ SUSPICIOUS (Medium Confidence)
            </p>
            <p>
              <span style={{ color: 'var(--danger)' }}>&lt; {thresholds.suspicious}%</span>
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
          className="px-8 py-3 rounded-lg font-bold transition-all duration-200"
          style={{ backgroundColor: 'var(--accent)', color: 'var(--bg)' }}
        >
          Try Another Verification
        </motion.button>
      </motion.div>
    </motion.div>
  )
}
