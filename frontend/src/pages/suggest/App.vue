<script setup lang="ts">

import {onMounted, ref} from "vue";

const songs = ref<string[]>([]);
const mood = ref<string>(window.location.search.split("=")[1]);

onMounted(async () => {
  // get query string that is called code
  // suggestions is an arrays of objects with the keys name, artist and id
  songs.value = await get_suggestions().then((res) => res.json()).then((data) => data.suggestions).then((suggestions) => suggestions.map((suggestion: any) => [suggestion.song, suggestion.artist, suggestion.id]));
});

async function get_suggestions() {
  return await fetch('/api/suggest', {
    method: 'POST',
    body: JSON.stringify({mood: mood.value}),
  });
}

async function create_playlist() {
  const playlist = await fetch('/api/create', {
    method: 'POST',
    body: JSON.stringify({ids: songs.value.map((song) => song[2])}),
  }).then((res) => res.json()).then((data) => data.playlist);
  // sleep for 1 second to let the playlist be created
  await new Promise((resolve) => setTimeout(resolve, 1000));
  window.location.href = playlist;
}

</script>

<template>
  <h1>I suggest you these songs:</h1>
  <h3 v-for="song in songs" :key="song">{{ song[0] }} - {{ song[1] }}</h3>
  <div class="spacer" />
  <button @click="create_playlist">Create a playlist</button>

</template>

<style scoped>
.spacer {
  height: 3em;
}
</style>
