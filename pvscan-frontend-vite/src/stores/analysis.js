import { defineStore } from 'pinia'
import axios from 'axios'


// filepath: /Users/jitendrajha/Desktop/PVSCAN/pvscan-frontend-vite/src/stores/analysis.js
const API_BASE = __API_BASE_URL__ || 'http://127.0.0.1:5001/'
console.log('API_BASE:', API_BASE)
  
export const useAnalysisStore = defineStore('analysis', {
  state: () => ({
    isLoading: false,
    error: null,
    singleResult: null,
    batchResults: null,
    efficiencyMetrics: null,
    uploadProgress: 0 
  }),

  getters: {
    hasResults: (state) => state.singleResult || state.batchResults,
    totalPanelsAnalyzed: (state) => {
      if (state.batchResults) {
        return state.batchResults.summary?.total_images || 0
      }
      return state.singleResult ? 1 : 0
    },
    averageScore: (state) => {
      if (state.batchResults) {
        return state.batchResults.summary?.average_score || 0
      }
      return state.singleResult?.total_score || 0
    }
  },

  actions: {
    async checkHealth() {
      try {
        const response = await axios.get(`${API_BASE}/api/health`)
        return response.data
      } catch (error) {
        this.error = 'Failed to connect to analysis service'
        throw error
      }
    },

    async analyzeSingleImage(file) {
      this.isLoading = true
      this.error = null
      this.uploadProgress = 0

      try {
        const formData = new FormData()
        formData.append('image', file)

        const response = await axios.post(`${API_BASE}/api/analyze-single`, formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          },
          onUploadProgress: (progressEvent) => {
            this.uploadProgress = Math.round(
              (progressEvent.loaded * 100) / progressEvent.total
            )
          }
        })

        this.singleResult = response.data
        this.batchResults = null // Clear batch results
        return response.data
      } catch (error) {
        this.error = error.response?.data?.error || 'Failed to analyze image'
        throw error
      } finally {
        this.isLoading = false
        this.uploadProgress = 0
      }
    },

    async analyzeBatchImages(zipFile) {
      this.isLoading = true
      this.error = null
      this.uploadProgress = 0

      try {
        const formData = new FormData()
        formData.append('zipfile', zipFile)

        const response = await axios.post(`${API_BASE}/api/analyze-batch`, formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          },
          onUploadProgress: (progressEvent) => {
            this.uploadProgress = Math.round(
              (progressEvent.loaded * 100) / progressEvent.total
            )
          }
        })

        this.batchResults = response.data
        this.singleResult = null // Clear single result
        
        // Calculate efficiency metrics
        if (response.data.results && response.data.results.length > 0) {
          await this.calculateEfficiencyMetrics(response.data.results)
        }
        
        return response.data
      } catch (error) {
        this.error = error.response?.data?.error || 'Failed to analyze batch'
        throw error
      } finally {
        this.isLoading = false
        this.uploadProgress = 0
      }
    },

    async calculateEfficiencyMetrics(results) {
      try {
        const response = await axios.post(`${API_BASE}/api/efficiency-metrics`, {
          results: results
        })
        
        this.efficiencyMetrics = response.data
        return response.data
      } catch (error) {
        console.error('Failed to calculate efficiency metrics:', error)
        // Don't throw error as this is not critical
      }
    },

    clearResults() {
      this.singleResult = null
      this.batchResults = null
      this.efficiencyMetrics = null
      this.error = null
    },

    clearError() {
      this.error = null
    }
  }
})

