import React from "react";
import { useNavigate } from "react-router-dom";
import "./HomePage.css";
import logo from "./assets/logo.png";

const HomePage = () => {
  const navigate = useNavigate();

  return (
    <div className="homepage-container">
      <div className="homepage-content">
        <div className="title-with-logo">
          <img src={logo} alt="Logo" className="logo-left" />
          <h1 className="title-text">
            <span className="highlight">Virtual</span> Vakil
          </h1>
        </div>

        <p className="homepage-tagline">Making Legal Documents Simple for Everyone</p>
        <p className="homepage-subtext">
          Your AI‑powered legal assistant — simplifying complex legal documents with ease and confidence.
        </p>

        <div className="homepage-buttons">
          <button onClick={() => navigate("/analyze")}>📄 Analyze Document</button>
          <button onClick={() => navigate("/chat")}>🤖 Chat with Advisor</button>
        </div>
      </div>
    </div>
  );
};

export default HomePage;
