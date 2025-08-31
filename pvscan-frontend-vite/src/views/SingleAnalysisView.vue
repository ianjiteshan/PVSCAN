<template>
  <div class="single-analysis-view">
    <div class="container">
      <div class="page-header fade-in">
        <h1 class="page-title">Single Panel Analysis</h1>
        <p class="page-description">
          Upload a single solar panel image for detailed AI-powered analysis and condition assessment.
        </p>
      </div>

      <div class="analysis-container">
        <!-- Upload Section -->
        <div class="upload-section card" v-if="!analysisStore.singleResult">
          <div class="upload-area" 
               :class="{ 'drag-over': isDragOver, 'uploading': analysisStore.isLoading }"
               @drop="handleDrop"
               @dragover.prevent="isDragOver = true"
               @dragleave="isDragOver = false"
               @click="triggerFileInput">
            
            <input ref="fileInput" 
                   type="file" 
                   accept="image/*" 
                   @change="handleFileSelect" 
                   style="display: none;">
            
            <div v-if="!analysisStore.isLoading" class="upload-content">
              <PhotoIcon class="upload-icon" />
              <h3 class="upload-title">Drop your image here or click to browse</h3>
              <p class="upload-subtitle">Supports JPG, PNG, WEBP formats</p>
              <button class="btn-primary">Choose Image</button>
            </div>

            <div v-else class="upload-loading">
              <div class="loading-spinner"></div>
              <p class="loading-text">Analyzing image...</p>
              <div class="progress-bar">
                <div class="progress-fill" :style="{ width: analysisStore.uploadProgress + '%' }"></div>
              </div>
              <p class="progress-text">{{ analysisStore.uploadProgress }}%</p>
            </div>
          </div>

          <div v-if="analysisStore.error" class="error-message">
            <ExclamationTriangleIcon class="error-icon" />
            <span>{{ analysisStore.error }}</span>
            <button @click="analysisStore.clearError()" class="error-close">×</button>
          </div>
        </div>

        <!-- Results Section -->
        <div v-if="analysisStore.singleResult" class="results-section">
          <div class="results-header">
            <h2 class="results-title">Analysis Results</h2>
            <button @click="startNewAnalysis" class="btn-secondary">
              <ArrowPathIcon class="btn-icon" />
              Analyze Another
            </button>
          </div>

          <div class="results-grid">
            <!-- Overall Score Card -->
            <div class="score-card card">
              <div class="score-header">
                <h3>Overall Condition Score</h3>
                <div class="score-badge" :class="getScoreClass(analysisStore.singleResult.total_score)">
                  {{ analysisStore.singleResult.condition }}
                </div>
              </div>
              <div class="score-display">
                <div class="score-circle">
                  <svg class="score-svg" viewBox="0 0 100 100">
                    <circle cx="50" cy="50" r="45" fill="none" stroke="#e5e7eb" stroke-width="8"/>
                    <circle cx="50" cy="50" r="45" fill="none" 
                            :stroke="getScoreColor(analysisStore.singleResult.total_score)"
                            stroke-width="8"
                            stroke-linecap="round"
                            :stroke-dasharray="circumference"
                            :stroke-dashoffset="circumference - (analysisStore.singleResult.total_score / 100) * circumference"
                            transform="rotate(-90 50 50)"/>
                  </svg>
                  <div class="score-text">
                    <span class="score-number">{{ formatScore(analysisStore.singleResult.total_score) }}</span>
                    <span class="score-unit">%</span>
                  </div>
                </div>
              </div>
            </div>

            <!-- Detailed Predictions -->
            <div class="predictions-card card">
              <h3 class="card-title">Detailed Analysis</h3>
              <div class="predictions-list">
                <div v-for="(score, label) in analysisStore.singleResult.predictions" 
                     :key="label" 
                     class="prediction-item">
                  <div class="prediction-header">
                    <span class="prediction-label">{{ formatLabel(label) }}</span>
                    <span class="prediction-score">{{ score.toFixed(1) }}%</span>
                  </div>
                  <div class="prediction-bar">
                    <div class="prediction-fill" 
                         :style="{ 
                           width: score + '%',
                           backgroundColor: getPredictionColor(label, score)
                         }"></div>
                  </div>
                </div>
              </div>
            </div>

            <!-- Suggestions -->
            <div class="suggestions-card card">
              <h3 class="card-title">
                <LightBulbIcon class="card-icon" />
                Maintenance Suggestions
              </h3>
              <div class="suggestions-list">
                <div v-for="(suggestion, index) in filteredSuggestions"
                     :key="index"
                     class="suggestion-item">
                  <CheckCircleIcon class="suggestion-icon" />
                  <div class="suggestion-content">
                    <span class="suggestion-text">{{ suggestion.description || suggestion.suggestion || suggestion }}</span>
                    <span v-if="suggestion.efficiency_loss" class="efficiency-loss"
                          :class="getEfficiencyClass(suggestion.efficiency_loss)">
                      {{ suggestion.efficiency_loss }}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { useAnalysisStore } from '../stores/analysis'
import { 
  PhotoIcon, 
  ExclamationTriangleIcon,
  ArrowPathIcon,
  LightBulbIcon,
  CheckCircleIcon
} from '@heroicons/vue/24/outline'

export default {
  name: 'SingleAnalysisView',
  components: {
    PhotoIcon,
    ExclamationTriangleIcon,
    ArrowPathIcon,
    LightBulbIcon,
    CheckCircleIcon
  },
  setup() {
    const analysisStore = useAnalysisStore()
    const fileInput = ref(null)
    const isDragOver = ref(false)

    const circumference = computed(() => 2 * Math.PI * 45)

    const filteredSuggestions = computed(() => {
      if (!analysisStore.singleResult?.suggestions?.suggestions) return []

      return analysisStore.singleResult.suggestions.suggestions.filter(suggestion => {
        const suggestionText = suggestion.description || suggestion.suggestion || suggestion.issue || suggestion
        // Filter out suggestions related to "Panel Detected"
        return !suggestionText.toLowerCase().includes('panel detected')
      })
    })

    const triggerFileInput = () => {
      if (!analysisStore.isLoading) {
        fileInput.value.click()
      }
    }

    const handleFileSelect = (event) => {
      const file = event.target.files[0]
      if (file) {
        analyzeImage(file)
      }
    }

    const handleDrop = (event) => {
      event.preventDefault()
      isDragOver.value = false
      
      const files = event.dataTransfer.files
      if (files.length > 0) {
        analyzeImage(files[0])
      }
    }

    const analyzeImage = async (file) => {
      try {
        await analysisStore.analyzeSingleImage(file)
      } catch (error) {
        console.error('Analysis failed:', error)
      }
    }

    const startNewAnalysis = () => {
      analysisStore.clearResults()
      if (fileInput.value) {
        fileInput.value.value = ''
      }
    }

    const getScoreClass = (score) => {
      if (score >= 90) return 'excellent'
      if (score >= 80) return 'good'
      if (score >= 70) return 'average'
      if (score >= 60) return 'poor'
      return 'critical'
    }

    const getScoreColor = (score) => {
      if (score >= 90) return '#10b981'
      if (score >= 80) return '#3b82f6'
      if (score >= 70) return '#f59e0b'
      if (score >= 60) return '#ef4444'
      return '#dc2626'
    }

    const getPredictionColor = (label, score) => {
      if (label === 'Clean Panel') {
        return score > 70 ? '#10b981' : '#ef4444'
      }
      return score > 30 ? '#ef4444' : '#10b981'
    }

    const formatLabel = (label) => {
      return label.replace(/([A-Z])/g, ' $1').trim()
    }

    const getEfficiencyClass = (efficiencyLoss) => {
      // Remove % sign and convert to number
      const value = parseFloat(efficiencyLoss.replace('%', ''))
      if (value >= 100) return 'critical'
      if (value >= 50) return 'high'
      if (value >= 20) return 'medium'
      return 'low'
    }

    const formatScore = (score) => {
      if (score === null || score === undefined) return '0.0'
      return parseFloat(score).toFixed(1)
    }

    return {
      analysisStore,
      fileInput,
      isDragOver,
      circumference,
      filteredSuggestions,
      triggerFileInput,
      handleFileSelect,
      handleDrop,
      startNewAnalysis,
      getScoreClass,
      getScoreColor,
      getPredictionColor,
      formatLabel,
      formatScore,
      getEfficiencyClass
    }
  }
}
</script>

<style scoped>
.single-analysis-view {
  min-height: 100vh;
  padding: 40px 24px;
}

.container {
  max-width: 1200px;
  margin: 0 auto;
}

.page-header {
  text-align: center;
  margin-bottom: 60px;
}

.page-title {
  font-size: 3rem;
  font-weight: 700;
  color: white;
  margin-bottom: 16px;
}

.page-description {
  font-size: 1.25rem;
  color: rgba(255, 255, 255, 0.8);
  max-width: 600px;
  margin: 0 auto;
}

.analysis-container {
  max-width: 800px;
  margin: 0 auto;
}

.upload-section {
  margin-bottom: 40px;
}

.upload-area {
  border: 2px dashed #cbd5e1;
  border-radius: 12px;
  padding: 60px 40px;
  text-align: center;
  cursor: pointer;
  transition: all 0.3s ease;
  background: rgba(255, 255, 255, 0.02);
}

.upload-area:hover,
.upload-area.drag-over {
  border-color: #667eea;
  background: rgba(102, 126, 234, 0.05);
}

.upload-area.uploading {
  cursor: not-allowed;
  border-color: #94a3b8;
}

.upload-icon {
  width: 64px;
  height: 64px;
  color: #94a3b8;
  margin: 0 auto 24px;
}

.upload-title {
  font-size: 1.5rem;
  font-weight: 600;
  color: #1f2937;
  margin-bottom: 8px;
}

.upload-subtitle {
  color: #64748b;
  margin-bottom: 24px;
}

.upload-loading {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 16px;
}

.loading-spinner {
  width: 48px;
  height: 48px;
  border: 4px solid #e5e7eb;
  border-top: 4px solid #667eea;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.loading-text {
  font-size: 1.125rem;
  font-weight: 600;
  color: #4f46e5;
}

.progress-text {
  font-size: 0.875rem;
  color: #64748b;
}

.error-message {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 16px;
  background: #fef2f2;
  border: 1px solid #fecaca;
  border-radius: 8px;
  color: #dc2626;
  margin-top: 16px;
}

.error-icon {
  width: 20px;
  height: 20px;
  flex-shrink: 0;
}

.error-close {
  background: none;
  border: none;
  font-size: 1.5rem;
  cursor: pointer;
  color: #dc2626;
  margin-left: auto;
}

.results-section {
  animation: fadeIn 0.6s ease-out;
}

.results-header {
  display: flex;
  justify-content: between;
  align-items: center;
  margin-bottom: 32px;
}

.results-title {
  font-size: 2rem;
  font-weight: 700;
  color: white;
}

.btn-icon {
  width: 16px;
  height: 16px;
}

.results-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 32px;
  max-width: 100%;
}

.score-card {
  text-align: center;
  padding: 32px;
  background: rgba(255, 255, 255, 0.95);
  border-radius: 16px;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

.score-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 40px;
  flex-wrap: wrap;
  gap: 16px;
}

.score-header h3 {
  font-size: 1.5rem;
  font-weight: 700;
  color: #1f2937;
  margin: 0;
}

.score-badge {
  padding: 8px 16px;
  border-radius: 24px;
  font-size: 0.875rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  white-space: nowrap;
}

.score-badge.excellent {
  background: linear-gradient(135deg, #d1fae5, #a7f3d0);
  color: #065f46;
  box-shadow: 0 2px 4px rgba(16, 185, 129, 0.2);
}

.score-badge.good {
  background: linear-gradient(135deg, #dbeafe, #bfdbfe);
  color: #1e40af;
  box-shadow: 0 2px 4px rgba(59, 130, 246, 0.2);
}

.score-badge.average {
  background: linear-gradient(135deg, #fef3c7, #fde68a);
  color: #92400e;
  box-shadow: 0 2px 4px rgba(245, 158, 11, 0.2);
}

.score-badge.poor {
  background: linear-gradient(135deg, #fee2e2, #fca5a5);
  color: #991b1b;
  box-shadow: 0 2px 4px rgba(239, 68, 68, 0.2);
}

.score-badge.critical {
  background: linear-gradient(135deg, #fecaca, #f87171);
  color: #7f1d1d;
  box-shadow: 0 2px 4px rgba(220, 38, 38, 0.2);
}

.score-display {
  display: flex;
  justify-content: center;
  margin: 20px 0;
}

.score-circle {
  position: relative;
  width: 180px;
  height: 180px;
  margin: 0 auto;
}

.score-svg {
  width: 100%;
  height: 100%;
  transform: rotate(-90deg);
}

.score-text {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  text-align: center;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}

.score-number {
  font-size: 2.5rem;
  font-weight: 800;
  color: #1f2937;
  line-height: 1;
  margin-bottom: 2px;
}

.score-unit {
  font-size: 1.25rem;
  color: #64748b;
  font-weight: 500;
  margin-top: 2px;
}

.card-title {
  font-size: 1.25rem;
  font-weight: 600;
  color: #1f2937;
  margin-bottom: 24px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.card-icon {
  width: 20px;
  height: 20px;
  color: #4f46e5;
}

.predictions-card {
  padding: 32px;
  background: rgba(255, 255, 255, 0.95);
  border-radius: 16px;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

.predictions-list {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.prediction-item {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.prediction-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.prediction-label {
  font-weight: 500;
  color: #374151;
}

.prediction-score {
  font-weight: 600;
  color: #1f2937;
}

.prediction-bar {
  height: 8px;
  background: #e5e7eb;
  border-radius: 4px;
  overflow: hidden;
}

.prediction-fill {
  height: 100%;
  border-radius: 4px;
  transition: width 0.6s ease;
}

.suggestions-card {
  padding: 32px;
  background: rgba(255, 255, 255, 0.95);
  border-radius: 16px;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

.suggestions-list {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.suggestion-item {
  display: flex;
  align-items: flex-start;
  gap: 16px;
  padding: 20px;
  background: linear-gradient(135deg, #f8fafc, #f1f5f9);
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
}

.suggestion-item::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  width: 4px;
  height: 100%;
  background: linear-gradient(135deg, #10b981, #059669);
  border-radius: 0 4px 4px 0;
}

.suggestion-item:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
  border-color: #cbd5e1;
}

.suggestion-icon {
  width: 24px;
  height: 24px;
  color: #10b981;
  flex-shrink: 0;
  margin-top: 2px;
  background: rgba(16, 185, 129, 0.1);
  padding: 6px;
  border-radius: 8px;
}

.suggestion-content {
  display: flex;
  flex-direction: column;
  gap: 8px;
  flex: 1;
}

.suggestion-text {
  font-size: 0.95rem;
  line-height: 1.6;
  color: #374151;
  font-weight: 500;
}

.efficiency-loss {
  font-size: 0.85rem;
  font-weight: 600;
  padding: 4px 8px;
  border-radius: 6px;
  align-self: flex-start;
  white-space: nowrap;
}

.efficiency-loss.critical {
  background: linear-gradient(135deg, #fee2e2, #fecaca);
  color: #991b1b;
}

.efficiency-loss.high {
  background: linear-gradient(135deg, #fef3c7, #fde68a);
  color: #92400e;
}

.efficiency-loss.medium {
  background: linear-gradient(135deg, #dbeafe, #bfdbfe);
  color: #1e40af;
}

.efficiency-loss.low {
  background: linear-gradient(135deg, #d1fae5, #a7f3d0);
  color: #065f46;
}

@media (max-width: 768px) {
  .page-title {
    font-size: 2rem;
  }

  .upload-area {
    padding: 40px 20px;
  }

  .results-header {
    flex-direction: column;
    gap: 16px;
    align-items: stretch;
  }

  .results-grid {
    gap: 24px;
  }

  .score-card,
  .predictions-card,
  .suggestions-card {
    padding: 24px;
  }

  .score-header {
    flex-direction: column;
    gap: 12px;
    text-align: center;
  }

  .score-header h3 {
    font-size: 1.25rem;
  }

  .score-circle {
    width: 140px;
    height: 140px;
  }

  .score-number {
    font-size: 2rem;
  }

  .score-unit {
    font-size: 1rem;
  }

  .suggestion-item {
    padding: 16px;
    gap: 12px;
  }

  .suggestion-icon {
    width: 20px;
    height: 20px;
    padding: 4px;
  }

  .suggestion-item span {
    font-size: 0.9rem;
    line-height: 1.5;
  }
}

@media (max-width: 480px) {
  .single-analysis-view {
    padding: 20px 16px;
  }

  .page-title {
    font-size: 1.75rem;
  }

  .page-description {
    font-size: 1rem;
  }

  .analysis-container {
    max-width: 100%;
  }

  .score-card,
  .predictions-card,
  .suggestions-card {
    padding: 20px;
  }

  .score-circle {
    width: 120px;
    height: 120px;
  }

  .score-number {
    font-size: 1.75rem;
  }

  .score-badge {
    padding: 6px 12px;
    font-size: 0.75rem;
  }

  .card-title {
    font-size: 1.125rem;
    margin-bottom: 20px;
  }

  .suggestion-item {
    flex-direction: column;
    align-items: flex-start;
    gap: 8px;
    padding: 14px;
  }

  .suggestion-icon {
    align-self: flex-start;
  }

  .suggestion-content {
    gap: 6px;
  }

  .suggestion-text {
    font-size: 0.9rem;
    line-height: 1.5;
  }

  .efficiency-loss {
    font-size: 0.8rem;
    padding: 3px 6px;
  }
}
</style>

