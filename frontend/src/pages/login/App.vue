<script setup lang="ts">
function login() {
  window.location.href = 'http://localhost:5000/api/authorize';
}
</script>

<template>
  <h1>Accedi con Spotify</h1>
  <div class="spacer" />
  <button @click="login">Accedi</button>
</template>

<style scoped>
.spacer {
  height: 3em;
}
</style>
