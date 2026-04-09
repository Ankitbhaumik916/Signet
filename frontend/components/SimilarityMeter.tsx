'use client'

import { motion } from 'framer-motion'
import { useEffect, useState } from 'react'

interface SimilarityMeterProps {
  score: number
  inferenceMode?: 'hybrid-neural' | 'classical-fallback' | string
  isVisible: boolean
}

/**
 * Animated circular progress gauge showing similarity score
 * Color-coded: Green (genuine) / Yellow (suspicious) / Red (forged)
 */
export default function SimilarityMeter({ score, inferenceMode = 'classical-fallback', isVisible }: SimilarityMeterProps) {
  const [animatedScore, setAnimatedScore] = useState(0)

  const thresholds = inferenceMode === 'hybrid-neural'
    ? { genuine: 90, suspicious: 80 }
    : { genuine: 93, suspicious: 82 }

  const getGaugeColor = (value: number) => {
    const percent = value * 100
    if (percent >= thresholds.genuine) return '#22c55e'
    if (percent >= thresholds.suspicious) return '#eab308'
    return '#ef4444'
  }

  const getLabel = (value: number) => {
    const percent = value * 100
    if (percent >= thresholds.genuine) return 'GENUINE'
    if (percent >= thresholds.suspicious) return 'SUSPICIOUS'
    return 'FORGED'
  }

  useEffect(() => {
    if (!isVisible) {
      setAnimatedScore(0)
      return
    }

    let animationFrameId: NodeJS.Timeout
    const startTime = Date.now()
    const duration = 1000 // 1 second animation

    const animate = () => {
      const elapsed = Date.now() - startTime
      const progress = Math.min(elapsed / duration, 1)
      
      // Easing function for smooth animation
      const easeOutQuad = 1 - Math.pow(1 - progress, 2)
      setAnimatedScore(score * easeOutQuad)

      if (progress < 1) {
        animationFrameId = setTimeout(animate, 16) // ~60fps
      }
    }

    animate()

    return () => clearTimeout(animationFrameId)
  }, [score, isVisible])

  const radius = 70
  const circumference = 2 * Math.PI * radius
  const strokeDashoffset = circumference - (animatedScore / 100) * circumference

  const percentage = Math.round(animatedScore)

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.8 }}
      animate={isVisible ? { opacity: 1, scale: 1 } : { opacity: 0, scale: 0.8 }}
      transition={{ duration: 0.5, ease: 'easeInOut' }}
      className="flex flex-col items-center justify-center gap-6"
    >
      {/* SVG Circular Gauge */}
      <div className="relative w-64 h-64 flex items-center justify-center">
        <svg className="w-full h-full transform -rotate-90" viewBox="0 0 200 200">
          {/* Background circle */}
          <circle
            cx="100"
            cy="100"
            r={radius}
            fill="none"
            stroke="rgba(51, 65, 85, 0.5)"
            strokeWidth="12"
          />
          
          {/* Progress circle */}
          <motion.circle
            cx="100"
            cy="100"
            r={radius}
            fill="none"
            stroke={getGaugeColor(animatedScore / 100)}
            strokeWidth="12"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
            transition={{ type: 'tween', duration: 0.05 }}
          />
        </svg>

        {/* Center content */}
        <motion.div
          className="absolute inset-0 flex flex-col items-center justify-center gap-2"
          initial={{ opacity: 0 }}
          animate={isVisible ? { opacity: 1 } : { opacity: 0 }}
          transition={{ delay: 0.3 }}
        >
          <motion.div
            className="text-5xl font-bold bg-gradient-to-r from-primary-400 to-primary-300 bg-clip-text text-transparent"
            key={percentage}
          >
            {percentage}%
          </motion.div>
          <div className="text-xs text-gray-400 font-semibold tracking-wider">
            SIMILARITY
          </div>
        </motion.div>
      </div>

      {/* Score Label */}
      <motion.div
        className={`badge ${
          percentage >= thresholds.genuine
            ? 'badge-success'
            : percentage >= thresholds.suspicious
              ? 'badge-warning'
              : 'badge-danger'
        }`}
        initial={{ y: 10, opacity: 0 }}
        animate={isVisible ? { y: 0, opacity: 1 } : { y: 10, opacity: 0 }}
        transition={{ delay: 0.5 }}
      >
        {getLabel(animatedScore / 100)}
      </motion.div>

      {/* Score thresholds info */}
      <motion.div
        className="text-xs text-gray-400 text-center space-y-1 mt-4"
        initial={{ opacity: 0 }}
        animate={isVisible ? { opacity: 1 } : { opacity: 0 }}
        transition={{ delay: 0.6 }}
      >
        <div>≥ {thresholds.genuine}%: Genuine (High)</div>
        <div>{thresholds.suspicious}-{thresholds.genuine - 1}%: Suspicious (Medium)</div>
        <div>&lt; {thresholds.suspicious}%: Forged (High)</div>
      </motion.div>
    </motion.div>
  )
}
