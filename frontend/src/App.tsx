import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import LandingPage from './pages/LandingPage';
import Dashboard from './pages/Dashboard';
import PredictPage from './pages/PredictPage';
import TradePage from './pages/TradePage';
import { AppProvider } from './utils/context';

function App() {
  return (
    <AppProvider>
      <Router>
        <div className="App">
          <Routes>
            <Route path="/" element={<LandingPage />} />
            <Route path="/analyze" element={<LandingPage />} />
            <Route path="/predict" element={<PredictPage />} />
            <Route path="/trade" element={<TradePage />} />
            <Route path="/dashboard/:ticker" element={<Dashboard />} />
            <Route path="/dashboard" element={<LandingPage />} />
            <Route path="*" element={<LandingPage />} />
          </Routes>
        </div>
      </Router>
    </AppProvider>
  );
}

export default App;
