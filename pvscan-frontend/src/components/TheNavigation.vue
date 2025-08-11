<template>
  <nav class="navbar">
    <div class="nav-container">
      <div class="nav-brand">
        <img src="/logo_comp.png" alt="PVSCAN" class="logo" />
        <span class="brand-text">PVSCAN</span>
      </div>
      
      <div class="nav-links">
        <router-link to="/" class="nav-link" :class="{ active: $route.name === 'Home' }">
          <HomeIcon class="nav-icon" />
          Home
        </router-link>
        <router-link to="/single" class="nav-link" :class="{ active: $route.name === 'SingleAnalysis' }">
          <PhotoIcon class="nav-icon" />
          Single Analysis
        </router-link>
        <router-link to="/batch" class="nav-link" :class="{ active: $route.name === 'BatchAnalysis' }">
          <FolderIcon class="nav-icon" />
          Batch Analysis
        </router-link>
        <router-link to="/dashboard" class="nav-link" :class="{ active: $route.name === 'Dashboard' }">
          <ChartBarIcon class="nav-icon" />
          Dashboard
        </router-link>
      </div>

      <!-- Mobile menu button -->
      <button @click="toggleMobileMenu" class="mobile-menu-btn">
        <Bars3Icon v-if="!mobileMenuOpen" class="menu-icon" />
        <XMarkIcon v-else class="menu-icon" />
      </button>
    </div>

    <!-- Mobile menu -->
    <div v-if="mobileMenuOpen" class="mobile-menu">
      <router-link to="/" class="mobile-nav-link" @click="closeMobileMenu">
        <HomeIcon class="nav-icon" />
        Home
      </router-link>
      <router-link to="/single" class="mobile-nav-link" @click="closeMobileMenu">
        <PhotoIcon class="nav-icon" />
        Single Analysis
      </router-link>
      <router-link to="/batch" class="mobile-nav-link" @click="closeMobileMenu">
        <FolderIcon class="nav-icon" />
        Batch Analysis
      </router-link>
      <router-link to="/dashboard" class="mobile-nav-link" @click="closeMobileMenu">
        <ChartBarIcon class="nav-icon" />
        Dashboard
      </router-link>
    </div>
  </nav>
</template>

<script>
import { ref } from 'vue'
import { 
  HomeIcon, 
  PhotoIcon, 
  FolderIcon, 
  ChartBarIcon,
  Bars3Icon,
  XMarkIcon
} from '@heroicons/vue/24/outline'

export default {
  name: 'TheNavigation',
  components: {
    HomeIcon,
    PhotoIcon,
    FolderIcon,
    ChartBarIcon,
    Bars3Icon,
    XMarkIcon
  },
  setup() {
    const mobileMenuOpen = ref(false)

    const toggleMobileMenu = () => {
      mobileMenuOpen.value = !mobileMenuOpen.value
    }

    const closeMobileMenu = () => {
      mobileMenuOpen.value = false
    }

    return {
      mobileMenuOpen,
      toggleMobileMenu,
      closeMobileMenu
    }
  }
}
</script>

<style scoped>
.navbar {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  z-index: 50;
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  border-bottom: 1px solid rgba(255, 255, 255, 0.18);
  box-shadow: 0 4px 32px rgba(31, 38, 135, 0.37);
}

.nav-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 24px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 80px;
}

.nav-brand {
  display: flex;
  align-items: center;
  gap: 12px;
}

.logo {
  height: 40px;
  width: auto;
}

.brand-text {
  font-size: 24px;
  font-weight: 700;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.nav-links {
  display: flex;
  align-items: center;
  gap: 32px;
}

.nav-link {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 16px;
  border-radius: 8px;
  text-decoration: none;
  color: #64748b;
  font-weight: 500;
  transition: all 0.3s ease;
}

.nav-link:hover {
  color: #4f46e5;
  background: rgba(79, 70, 229, 0.1);
}

.nav-link.active {
  color: #4f46e5;
  background: rgba(79, 70, 229, 0.15);
}

.nav-icon {
  width: 20px;
  height: 20px;
}

.mobile-menu-btn {
  display: none;
  background: none;
  border: none;
  cursor: pointer;
  padding: 8px;
  border-radius: 8px;
  transition: background 0.3s ease;
}

.mobile-menu-btn:hover {
  background: rgba(79, 70, 229, 0.1);
}

.menu-icon {
  width: 24px;
  height: 24px;
  color: #64748b;
}

.mobile-menu {
  display: none;
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  border-top: 1px solid rgba(255, 255, 255, 0.18);
  padding: 16px 24px;
}

.mobile-nav-link {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 16px;
  border-radius: 8px;
  text-decoration: none;
  color: #64748b;
  font-weight: 500;
  transition: all 0.3s ease;
  margin-bottom: 8px;
}

.mobile-nav-link:hover {
  color: #4f46e5;
  background: rgba(79, 70, 229, 0.1);
}

@media (max-width: 768px) {
  .nav-links {
    display: none;
  }

  .mobile-menu-btn {
    display: block;
  }

  .mobile-menu {
    display: block;
  }

  .nav-container {
    padding: 0 16px;
  }
}
</style>

