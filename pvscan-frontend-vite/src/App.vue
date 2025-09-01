<template>
  <div id="app">
    <TheNavigation />
    <main class="main-content">
      <router-view />
    </main>
    <footer>
      Built With ❤️ For A Sustainable Future - Anjitesh Shandilya
    </footer>
  </div>
</template>

<script>
import TheNavigation from './components/TheNavigation.vue'

export default {
  name: 'App',
  components: {
    TheNavigation
  }
}
</script>

<style scoped>
.main-content {
  min-height: calc(100vh - 80px);
  padding-top: 80px;
}

footer {
  text-align: center;
  padding: 20px;
  background-color: #f0f0f0;
  font-size: 14px;
  position: fixed;
  bottom: 0;
  left: 0;
  width: 100%;
}
</style>

