<script setup lang="ts">

import {onMounted, ref} from "vue";

const songs = ref<string[]>([]);
const mood = ref<string>("...");
const interval = ref<number>(0);

onMounted(async () => {
  // get query string that is called code
  const code = new URLSearchParams(window.location.search).get("code");
  songs.value = await fetch(`/api/recent?code=${code}`).then((res) => res.json()).then((data) => data.songs);
  interval.value = setInterval(loading, 1000);
  await get_mood().then((res) => res.json()).then((data) => {
    clearInterval(interval.value);
    mood.value = data.mood
  });
});

function loading() {
  // check if mood has . in it
  if (mood.value.includes(".")) {
    if (mood.value.length >= 3) {
      mood.value = ".";
    } else {
      mood.value += ".";
    }
  }
}

async function get_mood() {
  return await fetch('/api/mood', {
    method: 'POST',
    body: JSON.stringify({songs: songs.value}),
  })
}

function suggest() {
  window.location.href = 'http://localhost:5000/suggest';
}

</script>

<template>
  <h1>In the last 7 days you listened to:</h1>
  <h3 v-for="song in songs" :key="song">{{ song[0] }} - {{ song[1] }}</h3>
  <div class="spacer" />
  <h2>From the lyrics your mood seems to be</h2>
  <h1 class="upper">{{ mood }}</h1>
  <button @click="suggest">Suggest me other songs</button>

</template>

<style scoped>
.spacer {
  height: 2em;
}

.upper {
  text-transform: uppercase;
}
</style>
