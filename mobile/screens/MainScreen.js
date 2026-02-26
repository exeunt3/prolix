import React, { useMemo, useState } from 'react';
import { ActivityIndicator, Animated, Image, Pressable, StyleSheet, Text, View } from 'react-native';
import * as ImagePicker from 'expo-image-picker';

const API_URL = process.env.EXPO_PUBLIC_API_URL || 'http://localhost:8000';

async function toBase64(uri) {
  const response = await fetch(uri);
  const blob = await response.blob();
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(reader.result.split(',')[1]);
    reader.readAsDataURL(blob);
  });
}

export default function MainScreen() {
  const [imageUri, setImageUri] = useState(null);
  const [paragraph, setParagraph] = useState('');
  const [traceId, setTraceId] = useState(null);
  const [loading, setLoading] = useState(false);
  const fade = useMemo(() => new Animated.Value(0), []);

  const pickImage = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({ quality: 0.8, base64: false });
    if (!result.canceled) {
      setImageUri(result.assets[0].uri);
      setParagraph('');
      setTraceId(null);
    }
  };

  const generate = async (event) => {
    if (!imageUri) return;
    setLoading(true);
    const { locationX, locationY } = event.nativeEvent;
    const tapX = Math.max(0, Math.min(1, locationX / 300));
    const tapY = Math.max(0, Math.min(1, locationY / 300));
    const b64 = await toBase64(imageUri);
    const form = new FormData();
    form.append('tap_x', String(tapX));
    form.append('tap_y', String(tapY));
    form.append('image_b64', b64);
    const resp = await fetch(`${API_URL}/generate`, { method: 'POST', body: form });
    const data = await resp.json();
    setParagraph(data.paragraph_text);
    setTraceId(data.trace_id);
    Animated.timing(fade, { toValue: 1, duration: 500, useNativeDriver: true }).start();
    setLoading(false);
  };

  const deepen = async () => {
    if (!traceId) return;
    setLoading(true);
    const resp = await fetch(`${API_URL}/deepen`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ trace_id: traceId }),
    });
    const data = await resp.json();
    setParagraph(data.paragraph_text);
    setTraceId(data.trace_id);
    setLoading(false);
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Prolix</Text>
      <Pressable onPress={pickImage} style={styles.pickButton}><Text style={styles.pickText}>Select Photo</Text></Pressable>
      {imageUri && (
        <Pressable onPress={generate}>
          <Image source={{ uri: imageUri }} style={styles.image} />
        </Pressable>
      )}
      {loading && <ActivityIndicator color="#d7c9a5" />}
      {!!paragraph && (
        <Animated.View style={{ opacity: fade }}>
          <Text style={styles.paragraph}>{paragraph}</Text>
          <View style={styles.buttonRow}>
            <Pressable onPress={deepen} style={styles.secondaryButton}><Text style={styles.pickText}>Go Deeper</Text></Pressable>
            <Pressable onPress={() => setParagraph('')} style={styles.secondaryButton}><Text style={styles.pickText}>Return to Surface</Text></Pressable>
          </View>
        </Animated.View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 18, backgroundColor: '#0B0B0D' },
  title: { color: '#ece5d4', fontSize: 34, marginBottom: 14, fontFamily: 'serif' },
  pickButton: { padding: 12, backgroundColor: '#262428', borderRadius: 8, marginBottom: 12 },
  pickText: { color: '#ece5d4', textAlign: 'center' },
  image: { width: 300, height: 300, borderRadius: 10, marginBottom: 12 },
  paragraph: { color: '#ece5d4', lineHeight: 24, fontSize: 17, fontFamily: 'serif' },
  buttonRow: { flexDirection: 'row', gap: 10, marginTop: 12 },
  secondaryButton: { flex: 1, padding: 10, backgroundColor: '#1a1a1d', borderRadius: 8 },
});
