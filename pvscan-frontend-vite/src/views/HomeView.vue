<template>
  <div class="home-view">
    <!-- Hero Section -->
    <section class="hero">
      <div class="hero-content">
        <div class="hero-text fade-in">
          <h1 class="hero-title">
            Advanced Solar Panel
            <span class="gradient-text" data-text="Inspection & Analysis">Inspection & Analysis</span>
          </h1>
          <p class="hero-description">
            Harness the power of AI to detect issues, optimize performance, and maximize efficiency 
            of your solar panel installations with our cutting-edge computer vision technology.
          </p>
          <div class="hero-actions">
            <router-link to="/single" class="btn-primary">
              <PhotoIcon class="btn-icon" />
              Analyze Single Panel
            </router-link>
            <router-link to="/batch" class="btn-secondary">
              <FolderIcon class="btn-icon" />
              Batch Analysis
            </router-link>
          </div>
        </div>
        
        <div class="hero-visual slide-in">
          <div class="visual-card">
            <img src="/logo_phone.png" alt="PVSCAN Analysis" class="hero-image" />
            <div class="floating-stats">
              <div class="stat-item">
                <div class="stat-value">98.5%</div>
                <div class="stat-label">Accuracy</div>
              </div>
              <div class="stat-item">
                <div class="stat-value">&lt; 2s</div>
                <div class="stat-label">Analysis Time</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- Features Section -->
    <section class="features">
      <div class="container">
        <h2 class="section-title">Comprehensive Analysis Capabilities</h2>
        <div class="features-grid">
          <div class="feature-card" v-for="feature in features" :key="feature.id">
            <div class="feature-icon">
              <component :is="feature.icon" />
            </div>
            <h3 class="feature-title">{{ feature.title }}</h3>
            <p class="feature-description">{{ feature.description }}</p>
          </div>
        </div>
      </div>
    </section>

    <!-- Stats Section -->
    <section class="stats">
      <div class="container">
        <div class="stats-grid">
          <div class="stat-card" v-for="stat in stats" :key="stat.label">
            <div class="stat-number">{{ stat.value }}</div>
            <div class="stat-description">{{ stat.label }}</div>
          </div>
        </div>
      </div>
    </section>

    <!-- CTA Section -->
    <section class="cta">
      <div class="container">
        <div class="cta-content">
          <h2 class="cta-title">Ready to Optimize Your Solar Installation?</h2>
          <p class="cta-description">
            Start analyzing your solar panels today and discover insights that can improve efficiency and reduce maintenance costs.
          </p>
          <div class="cta-actions">
            <router-link to="/single" class="btn-primary">
              Get Started Now
            </router-link>
            <router-link to="/dashboard" class="btn-secondary">
              View Dashboard
            </router-link>
          </div>
        </div>
      </div>
    </section>
  </div>
</template>

<script>
import { 
  PhotoIcon, 
  FolderIcon,
  ShieldCheckIcon,
  BoltIcon,
  CloudIcon,
  BeakerIcon,
  CogIcon,
  ChartBarIcon
} from '@heroicons/vue/24/outline'

export default {
  name: 'HomeView',
  components: {
    PhotoIcon,
    FolderIcon,
    ShieldCheckIcon,
    BoltIcon,
    CloudIcon,
    BeakerIcon,
    CogIcon,
    ChartBarIcon
  },
  data() {
    return {
      features: [
        {
          id: 1,
          icon: ShieldCheckIcon,
          title: 'Physical Damage Detection',
          description: 'Identify cracks, chips, and structural damage that could impact panel performance.'
        },
        {
          id: 2,
          icon: BoltIcon,
          title: 'Electrical Issue Analysis',
          description: 'Detect electrical problems and connection issues before they cause system failures.'
        },
        {
          id: 3,
          icon: CloudIcon,
          title: 'Environmental Impact Assessment',
          description: 'Analyze snow coverage, water obstruction, and weather-related performance impacts.'
        },
        {
          id: 4,
          icon: BeakerIcon,
          title: 'Contamination Detection',
          description: 'Identify foreign particles, dirt accumulation, and cleaning requirements.'
        },
        {
          id: 5,
          icon: CogIcon,
          title: 'Automated Maintenance Suggestions',
          description: 'Receive actionable recommendations for optimal panel maintenance and care.'
        },
        {
          id: 6,
          icon: ChartBarIcon,
          title: 'Efficiency Metrics Dashboard',
          description: 'Track performance trends and efficiency metrics across your entire installation.'
        }
      ],
      stats: [
        { value: '8+', label: 'Detection Categories' },
        { value: '98.5%', label: 'Analysis Accuracy' },
        { value: '< 2s', label: 'Processing Time' },
        { value: '24/7', label: 'Availability' }
      ]
    }
  }
}
</script>

<style scoped>
.home-view {
  min-height: 100vh;
}

.hero {
  padding: 80px 24px;
  max-width: 1200px;
  margin: 0 auto;
}

.hero-content {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 80px;
  align-items: center;
}

.hero-title {
  font-size: 3.5rem;
  font-weight: 700;
  line-height: 1.1;
  margin-bottom: 24px;
  color: white;
}

.hero-description {
  font-size: 1.25rem;
  line-height: 1.6;
  color: rgba(255, 255, 255, 0.9);
  margin-bottom: 40px;
}

.hero-actions {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
}

.btn-icon {
  width: 20px;
  height: 20px;
}

.hero-visual {
  position: relative;
}

.visual-card {
  position: relative;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 24px;
  padding: 40px;
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.2);
}

.hero-image {
  width: 100%;
  height: auto;
  border-radius: 16px;
}

.floating-stats {
  position: absolute;
  top: -20px;
  right: -20px;
  display: flex;
  gap: 16px;
}

.stat-item {
  background: rgba(255, 255, 255, 0.95);
  padding: 16px;
  border-radius: 12px;
  text-align: center;
  box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
}

.stat-value {
  font-size: 1.5rem;
  font-weight: 700;
  color: #4f46e5;
}

.stat-label {
  font-size: 0.875rem;
  color: #64748b;
  margin-top: 4px;
}

.features {
  padding: 120px 24px;
  background: rgba(255, 255, 255, 0.05);
}

.container {
  max-width: 1200px;
  margin: 0 auto;
}

.section-title {
  font-size: 2.5rem;
  font-weight: 700;
  text-align: center;
  margin-bottom: 80px;
  color: white;
}

.features-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
  gap: 32px;
}

.feature-card {
  background: rgba(255, 255, 255, 0.95);
  padding: 32px;
  border-radius: 16px;
  text-align: center;
  transition: transform 0.3s ease, box-shadow 0.3s ease;
  box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
}

.feature-card:hover {
  transform: translateY(-8px);
  box-shadow: 0 16px 48px rgba(31, 38, 135, 0.5);
}

.feature-icon {
  width: 64px;
  height: 64px;
  margin: 0 auto 24px;
  padding: 16px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border-radius: 16px;
  color: white;
}

.feature-icon svg {
  width: 100%;
  height: 100%;
}

.feature-title {
  font-size: 1.5rem;
  font-weight: 600;
  margin-bottom: 16px;
  color: #1f2937;
}

.feature-description {
  color: #64748b;
  line-height: 1.6;
}

.stats {
  padding: 80px 24px;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 32px;
}

.stat-card {
  text-align: center;
  background: rgba(255, 255, 255, 0.1);
  padding: 40px 24px;
  border-radius: 16px;
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.2);
}

.stat-number {
  font-size: 3rem;
  font-weight: 700;
  color: white;
  margin-bottom: 8px;
}

.stat-description {
  color: rgba(255, 255, 255, 0.8);
  font-weight: 500;
}

.cta {
  padding: 120px 24px;
  background: rgba(255, 255, 255, 0.05);
}

.cta-content {
  text-align: center;
  max-width: 600px;
  margin: 0 auto;
}

.cta-title {
  font-size: 2.5rem;
  font-weight: 700;
  margin-bottom: 24px;
  color: white;
}

.cta-description {
  font-size: 1.25rem;
  color: rgba(255, 255, 255, 0.9);
  margin-bottom: 40px;
  line-height: 1.6;
}

.cta-actions {
  display: flex;
  gap: 16px;
  justify-content: center;
  flex-wrap: wrap;
}

@media (max-width: 768px) {
  .hero-content {
    grid-template-columns: 1fr;
    gap: 40px;
    text-align: center;
  }

  .hero-title {
    font-size: 2.5rem;
  }

  .floating-stats {
    position: static;
    justify-content: center;
    margin-top: 24px;
  }

  .features-grid {
    grid-template-columns: 1fr;
  }

  .stats-grid {
    grid-template-columns: repeat(2, 1fr);
  }

  .section-title,
  .cta-title {
    font-size: 2rem;
  }

  .cta-actions {
    flex-direction: column;
    align-items: center;
  }
}
</style>

