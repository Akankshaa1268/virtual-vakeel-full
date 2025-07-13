import React, { useState } from "react";
import "./Analyze.css";

const Analyze = () => {
  const [uploadedFile, setUploadedFile] = useState(null);
  const [summary, setSummary] = useState("");

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    setUploadedFile(file);

    // Placeholder summary - backend can later provide this
    setSummary("This is a placeholder summary for the uploaded legal document.");
  };

  const speakText = (text, lang = "hi-IN") => {
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = lang;
    window.speechSynthesis.speak(utterance);
  };

  const stopSpeaking = () => {
    window.speechSynthesis.cancel();
  };

  return (
    <div className="analyze-container">
      <h1>📄 Analyze Legal Document</h1>

      <input
        type="file"
        accept="image/*,application/pdf"
        onChange={handleFileChange}
        className="file-input"
      />

      {uploadedFile && (
        <div className="preview-section">
          <p><strong>File:</strong> {uploadedFile.name}</p>
        </div>
      )}

      {summary && (
        <div className="summary-section">
          <h2>Summary</h2>
          <p>{summary}</p>

          <div className="audio-controls">
            <button onClick={() => speakText(summary, "hi-IN")}>
              🔊 सुनो हिंदी
            </button>
            <button onClick={() => speakText(summary, "en-US")}>
              🔊 Listen English
            </button>
            <button onClick={stopSpeaking}>⛔ Stop</button>
          </div>
        </div>
      )}
    </div>
  );
 

};

export default Analyze;
