<template>
  <div id="app">
    <TheNavigation />
    <main class="main-content">
      <router-view />
    </main>
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
</style>

