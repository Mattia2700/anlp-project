<script setup lang="ts">
function login() {
  window.location.href = 'http://localhost:5000/api/authorize';
}
</script>

<template>
  <h1>Login with Spotify</h1>
  <div class="spacer" />
  <button @click="login">Login</button>
</template>

<style scoped>
.spacer {
  height: 3em;
}
</style>
