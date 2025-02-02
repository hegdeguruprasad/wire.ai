import React from "react";
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import HomePage from "./pages/HomePage"; // Landing Page
import RAGPage from "./pages/RAGPage"; // RAG Page


function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/RAG" element={<RAGPage />} />
      </Routes>
    </Router>
  );
}

export default App;
