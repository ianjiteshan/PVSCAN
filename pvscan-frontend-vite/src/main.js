import { createApp } from 'vue'
import { createPinia } from 'pinia'
import { createRouter, createWebHistory } from 'vue-router'
import App from './App.vue'
import './style.css'

// Import views
import HomeView from './views/HomeView.vue'
import SingleAnalysisView from './views/SingleAnalysisView.vue'
import BatchAnalysisView from './views/BatchAnalysisView.vue'
import EfficiencyDashboard from './views/EfficiencyDashboard.vue'

// Create router
const routes = [
  { path: '/', name: 'Home', component: HomeView },
  { path: '/single', name: 'SingleAnalysis', component: SingleAnalysisView },
  { path: '/batch', name: 'BatchAnalysis', component: BatchAnalysisView },
  { path: '/dashboard', name: 'Dashboard', component: EfficiencyDashboard }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

// Create app
const app = createApp(App)
const pinia = createPinia()

app.use(pinia)
app.use(router)
app.mount('#app')

