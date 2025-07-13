import React, { useState } from "react";
import "./CaseTracker.css";

const CaseTracker = () => {
  const [cases, setCases] = useState([]);
  const [form, setForm] = useState({ title: "", status: "", nextDate: "" });

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleAddCase = (e) => {
    e.preventDefault();
    setCases([...cases, form]);
    setForm({ title: "", status: "", nextDate: "" });
  };

  return (
    <div className="case-tracker">
      <h2>📂 Case Tracker</h2>

      <form className="case-form" onSubmit={handleAddCase}>
        <input
          name="title"
          placeholder="Case Title"
          value={form.title}
          onChange={handleChange}
          required
        />
        <input
          name="status"
          placeholder="Status"
          value={form.status}
          onChange={handleChange}
        />
        <input
          name="nextDate"
          type="date"
          placeholder="Next Hearing Date"
          value={form.nextDate}
          onChange={handleChange}
        />
        <button type="submit">Add Case</button>
      </form>

      <div className="case-list">
        {cases.map((c, idx) => (
          <div key={idx} className="case-card">
            <h4>{c.title}</h4>
            <p>Status: {c.status}</p>
            <p>Next Date: {c.nextDate}</p>
          </div>
        ))}
      </div>
    </div>
  );
};

export default CaseTracker;
