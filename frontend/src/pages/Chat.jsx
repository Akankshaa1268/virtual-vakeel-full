import { useState } from 'react';
import chatImage from '../assets/law-chat.png'; // update with your actual image file

export default function Chat() {
  const [msg, setMsg] = useState('');
  const [response, setResponse] = useState('');

  const handleAsk = () => {
    setResponse("This is where your legal assistant will answer!");
  };

  return (
    <div
      style={{
        backgroundColor: '#eef5ff',
        minHeight: '200vh',
        padding: '2rem',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'start',
      }}
    >
      <div
        style={{
          width: '100%',
          maxWidth: '600px',
          background: '#ffffff',
          padding: '2rem',
          borderRadius: '12px',
          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.1)',
        }}
      >
        <div style={{ textAlign: 'center' }}>
          <img
            src={chatImage}
            alt="Chatbot"
            style={{ width: '200px', marginBottom: '1rem' }}
          />
          <h2 style={{ marginBottom: '1rem' }}>👨‍⚖️ Talk to Your Legal Advisor</h2>
        </div>

        <input
          type="text"
          value={msg}
          onChange={(e) => setMsg(e.target.value)}
          placeholder="e.g. Can landlord evict me without notice?"
          style={{
            width: '100%',
            padding: '0.8rem',
            margin: '1rem 0',
            borderRadius: '8px',
            border: '1px solid #ccc',
            fontSize: '16px',
          }}
        />
        <button
          onClick={handleAsk}
          style={{
            width: '100%',
            padding: '0.6rem',
            backgroundColor: '#4b6fff',
            color: 'white',
            border: 'none',
            borderRadius: '6px',
            fontSize: '15px',
            cursor: 'pointer',
          }}
        >
          Ask
        </button>

        {response && (
          <div style={{ marginTop: '2rem' }}>
            <h4>📩 <strong>Response:</strong></h4>
            <p>{response}</p>
          </div>
        )}

        <div style={{ marginTop: '2rem' }}>
          <h4>💬 <strong>Try asking:</strong></h4>
          <ul>
            <li>What is the procedure for filing an FIR?</li>
            <li>Can police arrest without a warrant?</li>
            <li>How to apply for anticipatory bail?</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
