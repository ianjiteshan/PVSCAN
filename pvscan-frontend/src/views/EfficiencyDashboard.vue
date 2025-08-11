<template>
  <div class="dashboard-view">
    <div class="container">
      <div class="page-header fade-in">
        <h1 class="page-title">Efficiency Dashboard</h1>
        <p class="page-description">
          Comprehensive analytics and insights for your solar panel fleet performance and maintenance optimization.
        </p>
      </div>

      <!-- No Data State -->
      <div v-if="!hasAnalysisData" class="no-data-section">
        <div class="no-data-card card">
          <div class="no-data-content">
            <ChartBarIcon class="no-data-icon" />
            <h3 class="no-data-title">No Analysis Data Available</h3>
            <p class="no-data-description">
              Run a batch analysis to see comprehensive efficiency metrics and insights on this dashboard.
            </p>
            <div class="no-data-actions">
              <router-link to="/batch" class="btn-primary">
                <FolderIcon class="btn-icon" />
                Start Batch Analysis
              </router-link>
              <router-link to="/single" class="btn-secondary">
                <PhotoIcon class="btn-icon" />
                Single Analysis
              </router-link>
            </div>
          </div>
        </div>
      </div>

      <!-- Dashboard Content -->
      <div v-else class="dashboard-content">
        <!-- Key Metrics -->
        <div class="metrics-section">
          <h2 class="section-title">Key Performance Indicators</h2>
          <div class="metrics-grid">
            <div class="metric-card card">
              <div class="metric-icon">
                <PhotoIcon />
              </div>
              <div class="metric-content">
                <div class="metric-value">{{ analysisStore.efficiencyMetrics?.total_panels || 0 }}</div>
                <div class="metric-label">Total Panels Analyzed</div>
              </div>
            </div>

            <div class="metric-card card">
              <div class="metric-icon" :class="getEfficiencyClass()">
                <BoltIcon />
              </div>
              <div class="metric-content">
                <div class="metric-value">{{ analysisStore.efficiencyMetrics?.average_score || 0 }}%</div>
                <div class="metric-label">Average Efficiency Score</div>
              </div>
            </div>

            <div class="metric-card card">
              <div class="metric-icon" :class="getRatingClass()">
                <StarIcon />
              </div>
              <div class="metric-content">
                <div class="metric-value">{{ analysisStore.efficiencyMetrics?.efficiency_rating || 'N/A' }}</div>
                <div class="metric-label">Overall Rating</div>
              </div>
            </div>

            <div class="metric-card card">
              <div class="metric-icon warning">
                <ExclamationTriangleIcon />
              </div>
              <div class="metric-content">
                <div class="metric-value">{{ getMaintenanceCount() }}</div>
                <div class="metric-label">Panels Need Attention</div>
              </div>
            </div>
          </div>
        </div>

        <!-- Score Distribution Chart -->
        <div class="chart-section">
          <h2 class="section-title">Score Distribution Analysis</h2>
          <div class="chart-grid">
            <div class="chart-card card">
              <h3 class="chart-title">Condition Categories</h3>
              <div class="distribution-chart">
                <div v-for="(category, key) in analysisStore.efficiencyMetrics?.score_distribution" 
                     :key="key" 
                     class="distribution-item">
                  <div class="distribution-header">
                    <span class="distribution-label">{{ formatCategoryLabel(key) }}</span>
                    <span class="distribution-count">{{ category.count }} panels</span>
                  </div>
                  <div class="distribution-bar">
                    <div class="distribution-fill" 
                         :class="key"
                         :style="{ width: category.percentage + '%' }"></div>
                  </div>
                  <div class="distribution-percentage">{{ category.percentage }}%</div>
                </div>
              </div>
            </div>

            <div class="chart-card card">
              <h3 class="chart-title">Performance Overview</h3>
              <div class="performance-stats">
                <div class="stat-row">
                  <span class="stat-label">Highest Score</span>
                  <span class="stat-value excellent">{{ analysisStore.efficiencyMetrics?.max_score || 0 }}%</span>
                </div>
                <div class="stat-row">
                  <span class="stat-label">Median Score</span>
                  <span class="stat-value">{{ analysisStore.efficiencyMetrics?.median_score || 0 }}%</span>
                </div>
                <div class="stat-row">
                  <span class="stat-label">Lowest Score</span>
                  <span class="stat-value critical">{{ analysisStore.efficiencyMetrics?.min_score || 0 }}%</span>
                </div>
                <div class="stat-row">
                  <span class="stat-label">Score Range</span>
                  <span class="stat-value">{{ getScoreRange() }}</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Maintenance Priorities -->
        <div v-if="analysisStore.efficiencyMetrics?.maintenance_priority?.length > 0" class="maintenance-section">
          <h2 class="section-title">Maintenance Priorities</h2>
          <div class="maintenance-grid">
            <div v-for="priority in analysisStore.efficiencyMetrics.maintenance_priority" 
                 :key="priority.issue" 
                 class="maintenance-card card">
              <div class="maintenance-header">
                <div class="maintenance-issue">
                  <WrenchScrewdriverIcon class="maintenance-icon" />
                  <span class="issue-name">{{ priority.issue }}</span>
                </div>
                <div class="priority-badge" :class="priority.priority.toLowerCase()">
                  {{ priority.priority }}
                </div>
              </div>
              <div class="maintenance-stats">
                <div class="affected-panels">
                  <span class="affected-count">{{ priority.affected_panels }}</span>
                  <span class="affected-label">panels affected</span>
                </div>
                <div class="affected-percentage">
                  {{ priority.percentage }}% of fleet
                </div>
              </div>
              <div class="maintenance-progress">
                <div class="progress-bar">
                  <div class="progress-fill" 
                       :class="priority.priority.toLowerCase()"
                       :style="{ width: priority.percentage + '%' }"></div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Issue Analysis -->
        <div class="issues-section">
          <h2 class="section-title">Common Issues Analysis</h2>
          <div class="issues-grid">
            <div v-for="(count, issue) in analysisStore.efficiencyMetrics?.common_issues" 
                 :key="issue" 
                 class="issue-card card">
              <div class="issue-header">
                <component :is="getIssueIcon(issue)" class="issue-icon" />
                <h3 class="issue-title">{{ formatIssueLabel(issue) }}</h3>
              </div>
              <div class="issue-stats">
                <div class="issue-count">{{ count }}</div>
                <div class="issue-label">panels affected</div>
              </div>
              <div class="issue-severity" :class="getIssueSeverity(count)">
                {{ getIssueSeverityLabel(count) }}
              </div>
            </div>
          </div>
        </div>

        <!-- Recommendations -->
        <div class="recommendations-section">
          <h2 class="section-title">Optimization Recommendations</h2>
          <div class="recommendations-grid">
            <div class="recommendation-card card">
              <div class="recommendation-icon">
                <LightBulbIcon />
              </div>
              <div class="recommendation-content">
                <h3 class="recommendation-title">Immediate Actions</h3>
                <ul class="recommendation-list">
                  <li v-for="action in getImmediateActions()" :key="action">{{ action }}</li>
                </ul>
              </div>
            </div>

            <div class="recommendation-card card">
              <div class="recommendation-icon">
                <CalendarIcon />
              </div>
              <div class="recommendation-content">
                <h3 class="recommendation-title">Scheduled Maintenance</h3>
                <ul class="recommendation-list">
                  <li v-for="task in getScheduledTasks()" :key="task">{{ task }}</li>
                </ul>
              </div>
            </div>

            <div class="recommendation-card card">
              <div class="recommendation-icon">
                <TrendingUpIcon />
              </div>
              <div class="recommendation-content">
                <h3 class="recommendation-title">Performance Optimization</h3>
                <ul class="recommendation-list">
                  <li v-for="tip in getOptimizationTips()" :key="tip">{{ tip }}</li>
                </ul>
              </div>
            </div>
          </div>
        </div>

        <!-- Action Center -->
        <div class="action-section">
          <h2 class="section-title">Action Center</h2>
          <div class="action-grid">
            <div class="action-card card">
              <h3 class="action-title">Export Analysis Report</h3>
              <p class="action-description">Download a comprehensive PDF report with all analysis results and recommendations.</p>
              <button @click="exportReport" class="btn-primary">
                <DocumentArrowDownIcon class="btn-icon" />
                Export Report
              </button>
            </div>

            <div class="action-card card">
              <h3 class="action-title">Schedule New Analysis</h3>
              <p class="action-description">Set up automated analysis schedules for continuous monitoring of your solar fleet.</p>
              <button @click="scheduleAnalysis" class="btn-secondary">
                <ClockIcon class="btn-icon" />
                Schedule Analysis
              </button>
            </div>

            <div class="action-card card">
              <h3 class="action-title">View Detailed Results</h3>
              <p class="action-description">Access individual panel analysis results and detailed condition assessments.</p>
              <router-link to="/batch" class="btn-secondary">
                <EyeIcon class="btn-icon" />
                View Results
              </router-link>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { computed } from 'vue'
import { useAnalysisStore } from '../stores/analysis'
import { 
  ChartBarIcon,
  FolderIcon,
  PhotoIcon,
  BoltIcon,
  StarIcon,
  ExclamationTriangleIcon,
  WrenchScrewdriverIcon,
  LightBulbIcon,
  CalendarIcon,
  TrendingUpIcon,
  DocumentArrowDownIcon,
  ClockIcon,
  EyeIcon,
  ShieldCheckIcon,
  CloudIcon,
  BeakerIcon,
  CogIcon
} from '@heroicons/vue/24/outline'

export default {
  name: 'EfficiencyDashboard',
  components: {
    ChartBarIcon,
    FolderIcon,
    PhotoIcon,
    BoltIcon,
    StarIcon,
    ExclamationTriangleIcon,
    WrenchScrewdriverIcon,
    LightBulbIcon,
    CalendarIcon,
    TrendingUpIcon,
    DocumentArrowDownIcon,
    ClockIcon,
    EyeIcon,
    ShieldCheckIcon,
    CloudIcon,
    BeakerIcon,
    CogIcon
  },
  setup() {
    const analysisStore = useAnalysisStore()

    const hasAnalysisData = computed(() => {
      return analysisStore.efficiencyMetrics && analysisStore.batchResults
    })

    const getEfficiencyClass = () => {
      const score = analysisStore.efficiencyMetrics?.average_score || 0
      if (score >= 85) return 'excellent'
      if (score >= 75) return 'good'
      if (score >= 65) return 'average'
      return 'poor'
    }

    const getRatingClass = () => {
      const rating = analysisStore.efficiencyMetrics?.efficiency_rating || ''
      return rating.toLowerCase()
    }

    const getMaintenanceCount = () => {
      if (!analysisStore.efficiencyMetrics?.score_distribution) return 0
      const dist = analysisStore.efficiencyMetrics.score_distribution
      return (dist.poor?.count || 0) + (dist.critical?.count || 0)
    }

    const getScoreRange = () => {
      const min = analysisStore.efficiencyMetrics?.min_score || 0
      const max = analysisStore.efficiencyMetrics?.max_score || 0
      return `${max - min}%`
    }

    const formatCategoryLabel = (key) => {
      return key.charAt(0).toUpperCase() + key.slice(1)
    }

    const formatIssueLabel = (issue) => {
      return issue.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
    }

    const getIssueIcon = (issue) => {
      const iconMap = {
        physical_damage: ShieldCheckIcon,
        electrical_damage: BoltIcon,
        snow_covered: CloudIcon,
        water_obstruction: CloudIcon,
        contamination: BeakerIcon,
        bird_interference: CogIcon
      }
      return iconMap[issue] || ExclamationTriangleIcon
    }

    const getIssueSeverity = (count) => {
      const total = analysisStore.efficiencyMetrics?.total_panels || 1
      const percentage = (count / total) * 100
      if (percentage > 20) return 'high'
      if (percentage > 10) return 'medium'
      return 'low'
    }

    const getIssueSeverityLabel = (count) => {
      const severity = getIssueSeverity(count)
      return severity.charAt(0).toUpperCase() + severity.slice(1) + ' Impact'
    }

    const getImmediateActions = () => {
      const actions = []
      const metrics = analysisStore.efficiencyMetrics
      
      if (!metrics) return actions

      if (metrics.score_distribution?.critical?.count > 0) {
        actions.push(`Inspect ${metrics.score_distribution.critical.count} critically damaged panels immediately`)
      }
      
      if (metrics.common_issues?.electrical_damage > 0) {
        actions.push('Schedule electrical system inspection for safety')
      }
      
      if (metrics.common_issues?.physical_damage > 0) {
        actions.push('Assess structural integrity of damaged panels')
      }

      if (actions.length === 0) {
        actions.push('Continue regular monitoring schedule')
      }

      return actions
    }

    const getScheduledTasks = () => {
      const tasks = []
      const metrics = analysisStore.efficiencyMetrics
      
      if (!metrics) return tasks

      if (metrics.common_issues?.contamination > 0) {
        tasks.push('Schedule cleaning for contaminated panels')
      }
      
      if (metrics.common_issues?.snow_covered > 0) {
        tasks.push('Plan snow removal procedures')
      }
      
      if (metrics.score_distribution?.poor?.count > 0) {
        tasks.push('Plan maintenance for underperforming panels')
      }

      if (tasks.length === 0) {
        tasks.push('Maintain quarterly inspection schedule')
      }

      return tasks
    }

    const getOptimizationTips = () => {
      const tips = []
      const metrics = analysisStore.efficiencyMetrics
      
      if (!metrics) return tips

      const avgScore = metrics.average_score || 0
      
      if (avgScore < 80) {
        tips.push('Consider upgrading panels with consistently low scores')
      }
      
      if (metrics.common_issues?.bird_interference > 0) {
        tips.push('Install bird deterrent systems')
      }
      
      tips.push('Implement predictive maintenance based on analysis trends')
      tips.push('Monitor weather patterns for proactive maintenance')

      return tips
    }

    const exportReport = () => {
      // Create a comprehensive report
      const reportData = {
        title: 'Solar Panel Fleet Analysis Report',
        generated_date: new Date().toISOString(),
        summary: analysisStore.batchResults?.summary,
        efficiency_metrics: analysisStore.efficiencyMetrics,
        recommendations: {
          immediate_actions: getImmediateActions(),
          scheduled_tasks: getScheduledTasks(),
          optimization_tips: getOptimizationTips()
        }
      }
      
      const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `pvscan_efficiency_report_${new Date().toISOString().split('T')[0]}.json`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
    }

    const scheduleAnalysis = () => {
      alert('Scheduling feature coming soon! This would integrate with your maintenance management system.')
    }

    return {
      analysisStore,
      hasAnalysisData,
      getEfficiencyClass,
      getRatingClass,
      getMaintenanceCount,
      getScoreRange,
      formatCategoryLabel,
      formatIssueLabel,
      getIssueIcon,
      getIssueSeverity,
      getIssueSeverityLabel,
      getImmediateActions,
      getScheduledTasks,
      getOptimizationTips,
      exportReport,
      scheduleAnalysis
    }
  }
}
</script>

<style scoped>
.dashboard-view {
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

.no-data-section {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 400px;
}

.no-data-card {
  max-width: 500px;
  text-align: center;
  padding: 60px 40px;
}

.no-data-icon {
  width: 80px;
  height: 80px;
  color: #94a3b8;
  margin: 0 auto 24px;
}

.no-data-title {
  font-size: 1.5rem;
  font-weight: 600;
  color: #1f2937;
  margin-bottom: 16px;
}

.no-data-description {
  color: #64748b;
  margin-bottom: 32px;
  line-height: 1.6;
}

.no-data-actions {
  display: flex;
  gap: 16px;
  justify-content: center;
  flex-wrap: wrap;
}

.btn-icon {
  width: 16px;
  height: 16px;
}

.dashboard-content {
  animation: fadeIn 0.6s ease-out;
}

.section-title {
  font-size: 1.75rem;
  font-weight: 600;
  color: white;
  margin-bottom: 24px;
}

.metrics-section {
  margin-bottom: 60px;
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 24px;
}

.metric-card {
  display: flex;
  align-items: center;
  gap: 20px;
  padding: 24px;
}

.metric-icon {
  width: 56px;
  height: 56px;
  padding: 14px;
  border-radius: 12px;
  background: #f1f5f9;
  color: #64748b;
}

.metric-icon.excellent {
  background: #d1fae5;
  color: #065f46;
}

.metric-icon.good {
  background: #dbeafe;
  color: #1e40af;
}

.metric-icon.average {
  background: #fef3c7;
  color: #92400e;
}

.metric-icon.poor,
.metric-icon.critical {
  background: #fee2e2;
  color: #991b1b;
}

.metric-icon.warning {
  background: #fef3c7;
  color: #92400e;
}

.metric-icon.optimal {
  background: #d1fae5;
  color: #065f46;
}

.metric-icon.moderate {
  background: #fef3c7;
  color: #92400e;
}

.metric-value {
  font-size: 2rem;
  font-weight: 700;
  color: #1f2937;
}

.metric-label {
  color: #64748b;
  font-weight: 500;
}

.chart-section {
  margin-bottom: 60px;
}

.chart-grid {
  display: grid;
  grid-template-columns: 2fr 1fr;
  gap: 24px;
}

.chart-card {
  padding: 24px;
}

.chart-title {
  font-size: 1.25rem;
  font-weight: 600;
  color: #1f2937;
  margin-bottom: 24px;
}

.distribution-chart {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.distribution-item {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.distribution-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.distribution-label {
  font-weight: 600;
  color: #374151;
}

.distribution-count {
  color: #64748b;
  font-size: 0.875rem;
}

.distribution-bar {
  height: 12px;
  background: #e5e7eb;
  border-radius: 6px;
  overflow: hidden;
}

.distribution-fill {
  height: 100%;
  border-radius: 6px;
  transition: width 0.8s ease;
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

.distribution-percentage {
  font-weight: 600;
  color: #1f2937;
  font-size: 0.875rem;
}

.performance-stats {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.stat-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 0;
  border-bottom: 1px solid #f1f5f9;
}

.stat-row:last-child {
  border-bottom: none;
}

.stat-label {
  color: #64748b;
  font-weight: 500;
}

.stat-value {
  font-weight: 600;
  color: #1f2937;
}

.stat-value.excellent {
  color: #065f46;
}

.stat-value.critical {
  color: #991b1b;
}

.maintenance-section {
  margin-bottom: 60px;
}

.maintenance-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 24px;
}

.maintenance-card {
  padding: 24px;
}

.maintenance-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.maintenance-issue {
  display: flex;
  align-items: center;
  gap: 12px;
}

.maintenance-icon {
  width: 20px;
  height: 20px;
  color: #4f46e5;
}

.issue-name {
  font-weight: 600;
  color: #1f2937;
}

.priority-badge {
  padding: 4px 12px;
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

.maintenance-stats {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.affected-count {
  font-size: 1.5rem;
  font-weight: 700;
  color: #1f2937;
}

.affected-label {
  color: #64748b;
  font-size: 0.875rem;
}

.affected-percentage {
  color: #64748b;
  font-weight: 500;
}

.maintenance-progress {
  margin-top: 12px;
}

.progress-bar {
  height: 8px;
  background: #e5e7eb;
  border-radius: 4px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  border-radius: 4px;
  transition: width 0.6s ease;
}

.progress-fill.high {
  background: #ef4444;
}

.progress-fill.medium {
  background: #f59e0b;
}

.progress-fill.low {
  background: #10b981;
}

.issues-section {
  margin-bottom: 60px;
}

.issues-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 24px;
}

.issue-card {
  padding: 20px;
  text-align: center;
}

.issue-header {
  margin-bottom: 16px;
}

.issue-icon {
  width: 40px;
  height: 40px;
  color: #4f46e5;
  margin: 0 auto 12px;
}

.issue-title {
  font-size: 1rem;
  font-weight: 600;
  color: #1f2937;
}

.issue-stats {
  margin-bottom: 12px;
}

.issue-count {
  font-size: 2rem;
  font-weight: 700;
  color: #1f2937;
}

.issue-label {
  color: #64748b;
  font-size: 0.875rem;
}

.issue-severity {
  padding: 4px 12px;
  border-radius: 12px;
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
}

.issue-severity.high {
  background: #fee2e2;
  color: #991b1b;
}

.issue-severity.medium {
  background: #fef3c7;
  color: #92400e;
}

.issue-severity.low {
  background: #d1fae5;
  color: #065f46;
}

.recommendations-section {
  margin-bottom: 60px;
}

.recommendations-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 24px;
}

.recommendation-card {
  padding: 24px;
}

.recommendation-icon {
  width: 48px;
  height: 48px;
  padding: 12px;
  border-radius: 12px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  margin-bottom: 16px;
}

.recommendation-title {
  font-size: 1.125rem;
  font-weight: 600;
  color: #1f2937;
  margin-bottom: 16px;
}

.recommendation-list {
  list-style: none;
  padding: 0;
  margin: 0;
}

.recommendation-list li {
  padding: 8px 0;
  color: #64748b;
  line-height: 1.5;
  position: relative;
  padding-left: 20px;
}

.recommendation-list li:before {
  content: "•";
  color: #4f46e5;
  font-weight: bold;
  position: absolute;
  left: 0;
}

.action-section {
  margin-bottom: 40px;
}

.action-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 24px;
}

.action-card {
  padding: 24px;
  text-align: center;
}

.action-title {
  font-size: 1.125rem;
  font-weight: 600;
  color: #1f2937;
  margin-bottom: 12px;
}

.action-description {
  color: #64748b;
  margin-bottom: 20px;
  line-height: 1.5;
}

@media (max-width: 768px) {
  .page-title {
    font-size: 2rem;
  }

  .chart-grid {
    grid-template-columns: 1fr;
  }

  .metrics-grid {
    grid-template-columns: repeat(2, 1fr);
  }

  .maintenance-grid,
  .recommendations-grid,
  .action-grid {
    grid-template-columns: 1fr;
  }

  .issues-grid {
    grid-template-columns: repeat(2, 1fr);
  }

  .no-data-card {
    padding: 40px 20px;
  }

  .no-data-actions {
    flex-direction: column;
    align-items: center;
  }
}
</style>

