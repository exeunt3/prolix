import React from 'react';
import { SafeAreaView, StatusBar } from 'react-native';
import MainScreen from './screens/MainScreen';

export default function App() {
  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: '#0B0B0D' }}>
      <StatusBar barStyle="light-content" />
      <MainScreen />
    </SafeAreaView>
  );
}
