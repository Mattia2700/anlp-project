<script setup lang="ts">

import {onMounted, ref} from "vue";

const songs = ref<string[]>([]);

onMounted(async () => {
  // get query string that is called code
  const code = new URLSearchParams(window.location.search).get("code");
  songs.value = await fetch(`/api/recent?code=${code}`).then((res) => res.json()).then((data) => data.songs);
});

function suggest() {
  window.location.href = 'http://localhost:5000/suggest';
}

</script>

<template>
  <h1>Negli ultimi 7 giorni hai ascoltato:</h1>
  <h3 v-for="song in songs" :key="song">{{ song[0] }} - {{ song[1] }}</h3>
  <div class="spacer" />
  <h2>Dai testi il tuo mood sembra</h2>
  <h1 class="upper">neutrale</h1>
  <button @click="suggest">Suggeriscimi altre canzoni</button>

</template>

<style scoped>
.spacer {
  height: 3em;
}

.upper {
  text-transform: uppercase;
}
</style>
