<script setup lang="ts">

import {onMounted, ref} from "vue";

const songs = ref<string[]>([]);

onMounted(async () => {
  // get query string that is called code
  const code = new URLSearchParams(window.location.search).get("code");
  songs.value = await fetch(`/api/recent?code=${code}`).then((res) => res.json()).then((data) => data.songs);
});

</script>

<template>
  <h1>I suggest you these songs:</h1>
  <h3 v-for="song in songs" :key="song">{{ song[0] }} - {{ song[1] }}</h3>
  <div class="spacer" />
  <button>Create a playlist</button>

</template>

<style scoped>
.spacer {
  height: 3em;
}
</style>
