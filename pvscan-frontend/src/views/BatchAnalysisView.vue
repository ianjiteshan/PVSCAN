<template>
  <div class="batch-analysis-view">
    <div class="container">
      <div class="page-header fade-in">
        <h1 class="page-title">Batch Panel Analysis</h1>
        <p class="page-description">
          Upload a ZIP file containing multiple solar panel images for comprehensive batch analysis and fleet-wide insights.
        </p>
      </div>

      <div class="analysis-container">
        <!-- Upload Section -->
        <div class="upload-section card" v-if="!analysisStore.batchResults">
          <div class="upload-area" 
               :class="{ 'drag-over': isDragOver, 'uploading': analysisStore.isLoading }"
               @drop="handleDrop"
               @dragover.prevent="isDragOver = true"
               @dragleave="isDragOver = false"
               @click="triggerFileInput">
            
            <input ref="fileInput" 
                   type="file" 
                   accept=".zip" 
                   @change="handleFileSelect" 
                   style="display: none;">
            
            <div v-if="!analysisStore.isLoading" class="upload-content">
              <FolderIcon class="upload-icon" />
              <h3 class="upload-title">Drop your ZIP file here or click to browse</h3>
              <p class="upload-subtitle">ZIP file containing JPG, PNG, WEBP images</p>
              <button class="btn-primary">Choose ZIP File</button>
            </div>

            <div v-else class="upload-loading">
              <div class="loading-spinner"></div>
              <p class="loading-text">Processing batch analysis...</p>
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
        <div v-if="analysisStore.batchResults" class="results-section">
          <div class="results-header">
            <h2 class="results-title">Batch Analysis Results</h2>
            <div class="header-actions">
              <button @click="exportResults" class="btn-secondary">
                <DocumentArrowDownIcon class="btn-icon" />
                Export Results
              </button>
              <button @click="startNewAnalysis" class="btn-secondary">
                <ArrowPathIcon class="btn-icon" />
                Analyze Another
              </button>
            </div>
          </div>

          <!-- Summary Cards -->
          <div class="summary-grid">
            <div class="summary-card card">
              <div class="summary-icon">
                <PhotoIcon />
              </div>
              <div class="summary-content">
                <div class="summary-value">{{ analysisStore.batchResults.summary.total_images }}</div>
                <div class="summary-label">Total Images</div>
              </div>
            </div>

            <div class="summary-card card">
              <div class="summary-icon success">
                <CheckCircleIcon />
              </div>
              <div class="summary-content">
                <div class="summary-value">{{ analysisStore.batchResults.summary.successful_analyses }}</div>
                <div class="summary-label">Successful</div>
              </div>
            </div>

            <div class="summary-card card">
              <div class="summary-icon warning" v-if="analysisStore.batchResults.summary.failed_analyses > 0">
                <ExclamationTriangleIcon />
              </div>
              <div class="summary-icon success" v-else>
                <CheckCircleIcon />
              </div>
              <div class="summary-content">
                <div class="summary-value">{{ analysisStore.batchResults.summary.failed_analyses }}</div>
                <div class="summary-label">Failed</div>
              </div>
            </div>

            <div class="summary-card card">
              <div class="summary-icon" :class="getScoreClass(analysisStore.batchResults.summary.average_score)">
                <ChartBarIcon />
              </div>
              <div class="summary-content">
                <div class="summary-value">{{ analysisStore.batchResults.summary.average_score }}%</div>
                <div class="summary-label">Average Score</div>
              </div>
            </div>
          </div>

          <!-- Efficiency Metrics -->
          <div v-if="analysisStore.efficiencyMetrics" class="efficiency-section">
            <h3 class="section-title">Fleet Efficiency Overview</h3>
            <div class="efficiency-grid">
              <div class="efficiency-card card">
                <h4 class="card-title">Score Distribution</h4>
                <div class="distribution-chart">
                  <div v-for="(category, key) in analysisStore.efficiencyMetrics.score_distribution" 
                       :key="key" 
                       class="distribution-item">
                    <div class="distribution-bar">
                      <div class="distribution-fill" 
                           :class="key"
                           :style="{ width: category.percentage + '%' }"></div>
                    </div>
                    <div class="distribution-info">
                      <span class="distribution-label">{{ formatCategoryLabel(key) }}</span>
                      <span class="distribution-value">{{ category.count }} ({{ category.percentage }}%)</span>
                    </div>
                  </div>
                </div>
              </div>

              <div class="efficiency-card card">
                <h4 class="card-title">Efficiency Rating</h4>
                <div class="rating-display">
                  <div class="rating-circle" :class="analysisStore.efficiencyMetrics.efficiency_rating.toLowerCase()">
                    <span class="rating-text">{{ analysisStore.efficiencyMetrics.efficiency_rating }}</span>
                  </div>
                  <div class="rating-stats">
                    <div class="stat-item">
                      <span class="stat-label">Median Score</span>
                      <span class="stat-value">{{ analysisStore.efficiencyMetrics.median_score }}%</span>
                    </div>
                    <div class="stat-item">
                      <span class="stat-label">Score Range</span>
                      <span class="stat-value">{{ analysisStore.efficiencyMetrics.min_score }}% - {{ analysisStore.efficiencyMetrics.max_score }}%</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <!-- Maintenance Priorities -->
            <div v-if="analysisStore.efficiencyMetrics.maintenance_priority.length > 0" class="priorities-card card">
              <h4 class="card-title">
                <WrenchScrewdriverIcon class="card-icon" />
                Maintenance Priorities
              </h4>
              <div class="priorities-list">
                <div v-for="priority in analysisStore.efficiencyMetrics.maintenance_priority" 
                     :key="priority.issue" 
                     class="priority-item">
                  <div class="priority-header">
                    <span class="priority-issue">{{ priority.issue }}</span>
                    <span class="priority-badge" :class="priority.priority.toLowerCase()">
                      {{ priority.priority }}
                    </span>
                  </div>
                  <div class="priority-stats">
                    <span class="priority-affected">{{ priority.affected_panels }} panels affected ({{ priority.percentage }}%)</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- Individual Results -->
          <div class="individual-results">
            <h3 class="section-title">Individual Panel Results</h3>
            <div class="results-grid">
              <div v-for="(result, index) in analysisStore.batchResults.results" 
                   :key="index" 
                   class="result-card card">
                <div class="result-header">
                  <h4 class="result-filename">{{ result.filename }}</h4>
                  <div class="result-score" :class="getScoreClass(result.total_score)">
                    {{ result.total_score }}%
                  </div>
                </div>
                
                <div class="result-condition">
                  <span class="condition-badge" :class="getScoreClass(result.total_score)">
                    {{ result.condition }}
                  </span>
                </div>

                <div class="result-issues">
                  <div v-for="issue in getTopIssues(result.predictions)" 
                       :key="issue.label" 
                       class="issue-item">
                    <span class="issue-label">{{ formatLabel(issue.label) }}</span>
                    <span class="issue-score">{{ issue.score.toFixed(1) }}%</span>
                  </div>
                </div>

                <button @click="showResultDetails(result)" class="result-details-btn">
                  View Details
                </button>
              </div>
            </div>
          </div>

          <!-- Errors Section -->
          <div v-if="analysisStore.batchResults.errors.length > 0" class="errors-section">
            <h3 class="section-title">Processing Errors</h3>
            <div class="errors-list card">
              <div v-for="(error, index) in analysisStore.batchResults.errors" 
                   :key="index" 
                   class="error-item">
                <ExclamationTriangleIcon class="error-icon" />
                <span>{{ error }}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Result Details Modal -->
    <div v-if="selectedResult" class="modal-overlay" @click="closeModal">
      <div class="modal-content" @click.stop>
        <div class="modal-header">
          <h3>{{ selectedResult.filename }}</h3>
          <button @click="closeModal" class="modal-close">×</button>
        </div>
        <div class="modal-body">
          <div class="modal-score">
            <span class="modal-score-value">{{ selectedResult.total_score }}%</span>
            <span class="modal-score-condition" :class="getScoreClass(selectedResult.total_score)">
              {{ selectedResult.condition }}
            </span>
          </div>
          <div class="modal-predictions">
            <h4>Detailed Analysis</h4>
            <div v-for="(score, label) in selectedResult.predictions" 
                 :key="label" 
                 class="modal-prediction">
              <span class="modal-prediction-label">{{ formatLabel(label) }}</span>
              <span class="modal-prediction-score">{{ score.toFixed(1) }}%</span>
            </div>
          </div>
          <div class="modal-suggestions">
            <h4>Suggestions</h4>
            <ul>
              <li v-for="suggestion in selectedResult.suggestions" :key="suggestion">
                {{ suggestion }}
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref } from 'vue'
import { useAnalysisStore } from '../stores/analysis'
import { 
  FolderIcon,
  ExclamationTriangleIcon,
  ArrowPathIcon,
  DocumentArrowDownIcon,
  PhotoIcon,
  CheckCircleIcon,
  ChartBarIcon,
  WrenchScrewdriverIcon
} from '@heroicons/vue/24/outline'

export default {
  name: 'BatchAnalysisView',
  components: {
    FolderIcon,
    ExclamationTriangleIcon,
    ArrowPathIcon,
    DocumentArrowDownIcon,
    PhotoIcon,
    CheckCircleIcon,
    ChartBarIcon,
    WrenchScrewdriverIcon
  },
  setup() {
    const analysisStore = useAnalysisStore()
    const fileInput = ref(null)
    const isDragOver = ref(false)
    const selectedResult = ref(null)

    const triggerFileInput = () => {
      if (!analysisStore.isLoading) {
        fileInput.value.click()
      }
    }

    const handleFileSelect = (event) => {
      const file = event.target.files[0]
      if (file) {
        analyzeBatch(file)
      }
    }

    const handleDrop = (event) => {
      event.preventDefault()
      isDragOver.value = false
      
      const files = event.dataTransfer.files
      if (files.length > 0) {
        analyzeBatch(files[0])
      }
    }

    const analyzeBatch = async (file) => {
      try {
        await analysisStore.analyzeBatchImages(file)
      } catch (error) {
        console.error('Batch analysis failed:', error)
      }
    }

    const startNewAnalysis = () => {
      analysisStore.clearResults()
      if (fileInput.value) {
        fileInput.value.value = ''
      }
    }

    const exportResults = () => {
      const data = {
        summary: analysisStore.batchResults.summary,
        results: analysisStore.batchResults.results,
        efficiency_metrics: analysisStore.efficiencyMetrics,
        export_date: new Date().toISOString()
      }
      
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `pvscan_batch_results_${new Date().toISOString().split('T')[0]}.json`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
    }

    const getScoreClass = (score) => {
      if (score >= 90) return 'excellent'
      if (score >= 80) return 'good'
      if (score >= 70) return 'average'
      if (score >= 60) return 'poor'
      return 'critical'
    }

    const formatLabel = (label) => {
      return label.replace(/([A-Z])/g, ' $1').trim()
    }

    const formatCategoryLabel = (key) => {
      return key.charAt(0).toUpperCase() + key.slice(1)
    }

    const getTopIssues = (predictions) => {
      const issues = Object.entries(predictions)
        .filter(([label]) => label !== 'Panel Detected' && label !== 'Clean Panel')
        .map(([label, score]) => ({ label, score }))
        .sort((a, b) => b.score - a.score)
        .slice(0, 3)
      
      return issues
    }

    const showResultDetails = (result) => {
      selectedResult.value = result
    }

    const closeModal = () => {
      selectedResult.value = null
    }

    return {
      analysisStore,
      fileInput,
      isDragOver,
      selectedResult,
      triggerFileInput,
      handleFileSelect,
      handleDrop,
      startNewAnalysis,
      exportResults,
      getScoreClass,
      formatLabel,
      formatCategoryLabel,
      getTopIssues,
      showResultDetails,
      closeModal
    }
  }
}
</script>

<style scoped>
.batch-analysis-view {
  min-height: 100vh;
  padding: 40px 24px;
}

.container {
  max-width: 1400px;
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
  max-width: none;
}

.results-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 32px;
}

.results-title {
  font-size: 2rem;
  font-weight: 700;
  color: white;
}

.header-actions {
  display: flex;
  gap: 12px;
}

.btn-icon {
  width: 16px;
  height: 16px;
}

.summary-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 24px;
  margin-bottom: 40px;
}

.summary-card {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 20px;
}

.summary-icon {
  width: 48px;
  height: 48px;
  padding: 12px;
  border-radius: 12px;
  background: #f1f5f9;
  color: #64748b;
}

.summary-icon.success {
  background: #d1fae5;
  color: #065f46;
}

.summary-icon.warning {
  background: #fef3c7;
  color: #92400e;
}

.summary-icon.excellent {
  background: #d1fae5;
  color: #065f46;
}

.summary-icon.good {
  background: #dbeafe;
  color: #1e40af;
}

.summary-icon.average {
  background: #fef3c7;
  color: #92400e;
}

.summary-icon.poor,
.summary-icon.critical {
  background: #fee2e2;
  color: #991b1b;
}

.summary-value {
  font-size: 2rem;
  font-weight: 700;
  color: #1f2937;
}

.summary-label {
  color: #64748b;
  font-weight: 500;
}

.section-title {
  font-size: 1.5rem;
  font-weight: 600;
  color: white;
  margin-bottom: 24px;
}

.efficiency-section {
  margin-bottom: 40px;
}

.efficiency-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
  margin-bottom: 24px;
}

.efficiency-card {
  padding: 24px;
}

.card-title {
  font-size: 1.125rem;
  font-weight: 600;
  color: #1f2937;
  margin-bottom: 20px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.card-icon {
  width: 20px;
  height: 20px;
  color: #4f46e5;
}

.distribution-chart {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.distribution-item {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.distribution-bar {
  height: 8px;
  background: #e5e7eb;
  border-radius: 4px;
  overflow: hidden;
}

.distribution-fill {
  height: 100%;
  border-radius: 4px;
  transition: width 0.6s ease;
}

.distribution-fill.excellent {
  background: #10b981;
}

.distribution-fill.good {
  background: #3b82f6;
}

.distribution-fill.average {
  background: #f59e0b;
}

.distribution-fill.poor {
  background: #ef4444;
}

.distribution-fill.critical {
  background: #dc2626;
}

.distribution-info {
  display: flex;
  justify-content: space-between;
  font-size: 0.875rem;
}

.distribution-label {
  font-weight: 500;
  color: #374151;
}

.distribution-value {
  color: #64748b;
}

.rating-display {
  display: flex;
  align-items: center;
  gap: 24px;
}

.rating-circle {
  width: 80px;
  height: 80px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 700;
  font-size: 0.875rem;
  text-transform: uppercase;
}

.rating-circle.optimal {
  background: #d1fae5;
  color: #065f46;
}

.rating-circle.good {
  background: #dbeafe;
  color: #1e40af;
}

.rating-circle.moderate {
  background: #fef3c7;
  color: #92400e;
}

.rating-circle.poor,
.rating-circle.critical {
  background: #fee2e2;
  color: #991b1b;
}

.rating-stats {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.stat-item {
  display: flex;
  justify-content: space-between;
  gap: 16px;
}

.stat-label {
  color: #64748b;
  font-size: 0.875rem;
}

.stat-value {
  font-weight: 600;
  color: #1f2937;
  font-size: 0.875rem;
}

.priorities-card {
  padding: 24px;
}

.priorities-list {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.priority-item {
  padding: 16px;
  background: #f8fafc;
  border-radius: 8px;
}

.priority-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}

.priority-issue {
  font-weight: 600;
  color: #1f2937;
}

.priority-badge {
  padding: 4px 8px;
  border-radius: 12px;
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
}

.priority-badge.high {
  background: #fee2e2;
  color: #991b1b;
}

.priority-badge.medium {
  background: #fef3c7;
  color: #92400e;
}

.priority-badge.low {
  background: #d1fae5;
  color: #065f46;
}

.priority-affected {
  color: #64748b;
  font-size: 0.875rem;
}

.individual-results {
  margin-bottom: 40px;
}

.results-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 24px;
}

.result-card {
  padding: 20px;
}

.result-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 12px;
}

.result-filename {
  font-size: 1rem;
  font-weight: 600;
  color: #1f2937;
  flex: 1;
  margin-right: 12px;
  word-break: break-word;
}

.result-score {
  font-size: 1.25rem;
  font-weight: 700;
  padding: 4px 8px;
  border-radius: 6px;
  min-width: 60px;
  text-align: center;
}

.result-score.excellent {
  background: #d1fae5;
  color: #065f46;
}

.result-score.good {
  background: #dbeafe;
  color: #1e40af;
}

.result-score.average {
  background: #fef3c7;
  color: #92400e;
}

.result-score.poor,
.result-score.critical {
  background: #fee2e2;
  color: #991b1b;
}

.result-condition {
  margin-bottom: 16px;
}

.condition-badge {
  padding: 4px 12px;
  border-radius: 12px;
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
}

.condition-badge.excellent {
  background: #d1fae5;
  color: #065f46;
}

.condition-badge.good {
  background: #dbeafe;
  color: #1e40af;
}

.condition-badge.average {
  background: #fef3c7;
  color: #92400e;
}

.condition-badge.poor,
.condition-badge.critical {
  background: #fee2e2;
  color: #991b1b;
}

.result-issues {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-bottom: 16px;
}

.issue-item {
  display: flex;
  justify-content: space-between;
  font-size: 0.875rem;
}

.issue-label {
  color: #64748b;
}

.issue-score {
  font-weight: 600;
  color: #1f2937;
}

.result-details-btn {
  width: 100%;
  padding: 8px 16px;
  background: #f1f5f9;
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  color: #4f46e5;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
}

.result-details-btn:hover {
  background: #e2e8f0;
}

.errors-section {
  margin-bottom: 40px;
}

.errors-list {
  padding: 20px;
}

.error-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px;
  background: #fef2f2;
  border-radius: 8px;
  color: #dc2626;
  margin-bottom: 8px;
}

.error-item:last-child {
  margin-bottom: 0;
}

.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  padding: 20px;
}

.modal-content {
  background: white;
  border-radius: 12px;
  max-width: 500px;
  width: 100%;
  max-height: 80vh;
  overflow-y: auto;
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px;
  border-bottom: 1px solid #e5e7eb;
}

.modal-header h3 {
  font-size: 1.25rem;
  font-weight: 600;
  color: #1f2937;
  margin: 0;
}

.modal-close {
  background: none;
  border: none;
  font-size: 1.5rem;
  cursor: pointer;
  color: #64748b;
  padding: 4px;
}

.modal-body {
  padding: 20px;
}

.modal-score {
  text-align: center;
  margin-bottom: 24px;
}

.modal-score-value {
  font-size: 2.5rem;
  font-weight: 700;
  color: #1f2937;
  display: block;
}

.modal-score-condition {
  padding: 4px 12px;
  border-radius: 12px;
  font-size: 0.875rem;
  font-weight: 600;
  text-transform: uppercase;
  margin-top: 8px;
  display: inline-block;
}

.modal-predictions {
  margin-bottom: 24px;
}

.modal-predictions h4 {
  font-size: 1.125rem;
  font-weight: 600;
  color: #1f2937;
  margin-bottom: 16px;
}

.modal-prediction {
  display: flex;
  justify-content: space-between;
  padding: 8px 0;
  border-bottom: 1px solid #f1f5f9;
}

.modal-prediction:last-child {
  border-bottom: none;
}

.modal-prediction-label {
  color: #64748b;
}

.modal-prediction-score {
  font-weight: 600;
  color: #1f2937;
}

.modal-suggestions h4 {
  font-size: 1.125rem;
  font-weight: 600;
  color: #1f2937;
  margin-bottom: 16px;
}

.modal-suggestions ul {
  list-style: none;
  padding: 0;
  margin: 0;
}

.modal-suggestions li {
  padding: 8px 0;
  color: #64748b;
  line-height: 1.5;
}

.modal-suggestions li:before {
  content: "•";
  color: #4f46e5;
  font-weight: bold;
  display: inline-block;
  width: 1em;
  margin-left: -1em;
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

  .header-actions {
    justify-content: center;
  }

  .summary-grid {
    grid-template-columns: repeat(2, 1fr);
  }

  .efficiency-grid {
    grid-template-columns: 1fr;
  }

  .results-grid {
    grid-template-columns: 1fr;
  }

  .rating-display {
    flex-direction: column;
    text-align: center;
    gap: 16px;
  }

  .modal-content {
    margin: 20px;
    max-height: calc(100vh - 40px);
  }
}
</style>

